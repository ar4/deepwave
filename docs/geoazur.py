"""HTTPS access to Geoazur's original Marmousi data.

Geoazur currently needs TLS 1.2 RSA key exchange, which CPython's default
cipher list excludes. Add only its AES-GCM suites, only for this host.
Certificate/hostname verification stays enabled; Geoazur requires TLS 1.2+.
Remove this compatibility setting when Geoazur supports modern key exchange.
"""

import http.client
import re
import shutil
import ssl
import urllib.parse
import urllib.request

MARMOUSI_PAGE = "https://www.geoazur.fr/WIND/bin/view/Main/Data/Marmousi"


def tls_context(url):
    """Return verified TLS settings, extending ciphers only for Geoazur."""
    context = ssl.create_default_context()
    parts = urllib.parse.urlsplit(url)
    if (
        parts.scheme == "https"
        and parts.hostname == "www.geoazur.fr"
        and parts.port in (None, 443)
    ):
        context.minimum_version = max(
            context.minimum_version, ssl.TLSVersion.TLSv1_2
        )
        ciphers = ":".join(c["name"] for c in context.get_ciphers())
        context.set_ciphers(
            ciphers + ":AES256-GCM-SHA384:AES128-GCM-SHA256"
        )
    return context


class _HTTPSHandler(urllib.request.HTTPSHandler):
    """Select TLS settings per request, including after redirects."""

    def https_open(self, request):
        """Open HTTPS without carrying Geoazur's settings to other hosts."""
        return self.do_open(
            http.client.HTTPSConnection,
            request,
            context=tls_context(request.full_url),
        )


class _HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    """Reject redirects that downgrade a verified HTTPS download."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Follow redirects only when the destination still uses HTTPS."""
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError(f"Refusing non-HTTPS redirect to {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_https(url):
    """Open a verified HTTPS response with a bounded connection timeout."""
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError(f"Expected an HTTPS URL, got {url}")
    opener = urllib.request.build_opener(
        _HTTPSHandler(), _HTTPSOnlyRedirect()
    )
    return opener.open(url, timeout=30)


def download(url, destination):
    """Stream HTTPS to disk; callers verify the model checksum."""
    with open_https(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def check_marmousi_link(app):
    """Require a successful Geoazur check during every Sphinx link build."""
    if app.builder.name != "linkcheck":
        return
    from sphinx.errors import ExtensionError
    from sphinx.util import logging

    try:
        with open_https(MARMOUSI_PAGE) as response:
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}")
    except (OSError, ValueError) as error:
        raise ExtensionError(
            f"Geoazur link check failed for {MARMOUSI_PAGE}: {error}"
        ) from error
    logging.getLogger(__name__).info(
        "Geoazur link verified (certificate checked): %s", MARMOUSI_PAGE
    )
    # Only this checked URL bypasses Sphinx's generic transport.
    # Do not suppress failures or exempt other Geoazur URLs/anchors.
    app.config.linkcheck_ignore = [
        *app.config.linkcheck_ignore,
        "^" + re.escape(MARMOUSI_PAGE) + "$",
    ]

# Nix derivation made to mirror the original hosted at
# https://github.com/NixOS/nixpkgs/blob/nixos-unstable/pkgs/development/python-modules/deepwave/default.nix
# to use it for a development environment run
#
# > $ nix-shell
#
# at the root of the repository. To test the build run:
#
# > $ nix-build

with import <nixpkgs> { };

python3Packages.buildPythonPackage rec {
  pname = "deepwave";
  version = "0.0.19.dev1";
  format = "pyproject";

  src = ./.;

  propagatedBuildInputs = with python3Packages; [ torch ];

  nativeCheckInputs = [
    which
    python3Packages.scipy
    python3Packages.pytest-xdist
    python3Packages.pytestCheckHook
  ];

  pythonImportsCheck = [ "deepwave" ];

  meta = with lib; {
    description = "Wave propagation modules for PyTorch";
    homepage = "https://github.com/ar4/deepwave";
    license = licenses.mit;
    platforms = intersectLists platforms.x86_64 platforms.linux;
  };
}

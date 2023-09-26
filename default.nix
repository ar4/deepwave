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

let
  linePatch = ''
    import os
    os.environ['PATH'] = os.environ['PATH'] + ':${ninja}/bin'
  '';
in
python3Packages.buildPythonPackage rec {
  pname = "deepwave";
  version = "0.0.20";
  format = "pyproject";

  src = ./.;

  # unable to find ninja although it is available, most likely because it looks for its pip version
  postPatch = ''
    substituteInPlace setup.cfg --replace "ninja" ""

    # Adding ninja to the path forcibly
    mv src/deepwave/__init__.py tmp
    echo "${linePatch}" > src/deepwave/__init__.py
    cat tmp >> src/deepwave/__init__.py
    rm tmp
  '';

  # The source files are compiled at runtime and cached at the
  # $HOME/.cache folder, so for the check phase it is needed to
  # have a temporary home. This is also the reason ninja is not
  # needed at the nativeBuildInputs, since it will only be used
  # at runtime.
  preBuild = ''
    export HOME=$(mktemp -d)
  '';

  propagatedBuildInputs = with python3Packages; [ torch pybind11 ];

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

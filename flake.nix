{
  description = "AURORA Drone ML Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python313
            pkgs.python313Packages.pip
            pkgs.python313Packages.virtualenv
            pkgs.libcamera
            pkgs.i2c-tools
            pkgs.bluez
            pkgs.bluez-tools
            pkgs.gcc
            pkgs.cmake
          ];

          shellHook = ''
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              python3 -m venv .venv
              source .venv/bin/activate
              pip install -r requirements_project.txt
            else
              source .venv/bin/activate
            fi
            echo "AURORA environment ready"
            echo "Run: python3 CNN.py"
          '';
        };
      });
}

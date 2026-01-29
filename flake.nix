{
  description = "My flake for this project";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ ];
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];
      perSystem =
        { pkgs, ... }:
        {
          devShells.default = pkgs.mkShell {
            packages =
              (with pkgs.python313Packages; [
                sentence-transformers
                torch
              ])
              ++ (with pkgs; [
                # Rust toolchain
                rustc
                rust-analyzer
                cargo

                # For openssl-sys crate
                openssl.dev
                pkg-config

                # For gRPC/protobuf
                protobuf

                grpcurl
              ]);

            env = {
              PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
              VK_ICD_FILENAMES = "/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json";
              LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib";
            };
          };
        };
      flake = { };
    };
}

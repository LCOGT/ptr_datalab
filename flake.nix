{
  description = "Datalab";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    devenv-k8s.url = "github:LCOGT/devenv-k8s";

    nixpkgs.follows = "devenv-k8s/nixpkgs";
    flake-parts.follows = "devenv-k8s/flake-parts";
  };

  nixConfig = {
    extra-substituters = [
      "https://devenv.cachix.org"
      "https://lco-public.cachix.org"
    ];

    extra-trusted-public-keys = [
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw="
      "lco-public.cachix.org-1:zSmLK7CkAehZ7QzTLZKt+5Y26Lr0w885GUB4GlT1SCg="
    ];
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devenv-k8s.flakeModules.default
      ];

      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];

      perSystem = { config, self', inputs', pkgs, system, ... }: {
        # Per-system attributes can be defined here. The self' and inputs'
        # module parameters provide easy access to attributes of the same
        # system.

        # https://devenv.sh/basics/
        # Enter using `nix develop --impure`
        config.devenv.shells.default = {

          # https://devenv.sh/packages/
          packages = [

          ];

          # https://devenv.sh/reference/options/#entershell
          enterShell = ''
            export KUBECONFIG="`pwd`/local-kubeconfig"

            echo "Setting KUBECONFIG=$KUBECONFIG"
            echo "If you would like to use a local K8s cluster across multiple projects, then set 'KUBECONFIG' to a common path"
            echo "in both projects before running the ctlptl command to create the local cluster."
            echo
            echo "Starting local registry and cluster"
            ctlptl apply -f local-registry.yaml -f local-cluster.yaml
            echo
            echo "Starting Dependencies in dev mode"
            skaffold run -m deps --tail=false --verbosity=warn
            echo "Ensure secrets.env.changeme is properly configured. Then run: "
            echo "skaffold dev -m app --port-forward"
            echo
          '';
        };
      };

      flake = {
        # The usual flake attributes can be defined here, including system-
        # agnostic ones like nixosModule and system-enumerating ones, although
        # those are more easily expressed in perSystem.

      };
    };
}

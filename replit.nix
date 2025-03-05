
{ pkgs }: {
  deps = [
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
    pkgs.git-lfs
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
  ];
}

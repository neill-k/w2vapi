
{ pkgs }: {
  deps = [
    pkgs.git-lfs
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
  ];
}

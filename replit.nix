{ pkgs }: {
  deps = [
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.python39
    pkgs.python39Packages.pip
    pkgs.python39Packages.numpy
    pkgs.python39Packages.gensim
  ];
} 
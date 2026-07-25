cask "moregpu" do
  version "0.3.0"
  sha256 "da6a5bc4175283bce0b1874052c249181ca8cd2073f19c2d440284da8890c9d4"

  url "https://github.com/ArioMoniri/moregpu/releases/download/v#{version}/moregpu"
  name "MoreGPU"
  desc "CLI for the MoreGPU native GPU compute pool (serve / join / control / monitor)"
  homepage "https://github.com/ArioMoniri/moregpu"

  depends_on formula: "deno"

  binary "moregpu"

  caveats <<~EOS
    MoreGPU runs its coordinator and worker via Deno (installed as a dependency).
    Run `moregpu` on its own for the interactive menu, or `moregpu serve --worker`
    to start a pool that also lends this machine.
  EOS
end

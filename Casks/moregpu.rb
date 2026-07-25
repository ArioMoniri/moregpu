cask "moregpu" do
  version "0.3.0"
  sha256 "e099af466bad747d7c85e6fd881ffcc9881877ea50224377383f362152579920"

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

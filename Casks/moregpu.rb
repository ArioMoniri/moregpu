cask "moregpu" do
  version "0.4.0"
  sha256 "0b267881d394184290bca86751c3043d45a9ad8ccb949e2bb03277c05fb30f00"

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

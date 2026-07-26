cask "moregpu" do
  version "0.6.0"
  sha256 "06f817950aea18ff09eec0cb896fcbce1b7bc7a6849cac41a925605ebc0e2ee0"

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

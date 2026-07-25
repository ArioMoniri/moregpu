cask "moregpu" do
  version "0.5.0"
  sha256 "88cc6c39e0b40c84eb7cce5fb31cd97e39741890604667533e645ab83abfb2d1"

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

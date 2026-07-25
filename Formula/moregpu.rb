# Homebrew formula for the MoreGPU CLI.
#
#   brew tap ariomoniri/moregpu https://github.com/ArioMoniri/moregpu
#   brew install --HEAD moregpu
#
# (HEAD install tracks main; tagged bottles can be added later via a `url` + `sha256`.)
class Moregpu < Formula
  desc "Run, stop, and monitor a MoreGPU native GPU compute pool"
  homepage "https://github.com/ArioMoniri/moregpu"
  license "Apache-2.0"
  head "https://github.com/ArioMoniri/moregpu.git", branch: "main"

  depends_on "deno"

  def install
    bin.install "scripts/moregpu"
  end

  test do
    assert_match "moregpu CLI", shell_output("#{bin}/moregpu version")
  end
end

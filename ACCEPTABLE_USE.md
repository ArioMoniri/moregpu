# Acceptable Use Policy

**Project:** MoreGPU — a native GPU compute pool
**Maintainer:** Ariorad Moniri
**Repository:** https://github.com/ArioMoniri/moregpu
**License:** Apache-2.0 (see [`LICENSE`](./LICENSE))

MoreGPU is neutral infrastructure: an admin runs a coordinator/admin server, and
worker machines make an **outbound** WebSocket connection to join that pool and
contribute GPU or CPU compute. Like any general-purpose compute tool, MoreGPU can
be used lawfully or unlawfully. This Acceptable Use Policy ("AUP") explains the
boundaries. **You are solely responsible for how you deploy and operate it.**

This AUP supplements, and does not limit, the disclaimers and limitation of
liability in the Apache-2.0 license. If any term here appears to conflict with
that license, the license controls as to warranty and liability.

---

## 1. Authorized machines only

You may run a MoreGPU worker, or point a worker at an admin server, **only on
machines and networks that you own or that you are explicitly authorized by the
owner to use for this purpose.**

"Explicitly authorized" means you have the owner's clear permission to run
background compute, GPU/CPU-intensive, and long-running non-interactive
workloads on that machine. Being *able* to log in to a machine, or having it
assigned to you at work or school, is **not** the same as being authorized to
enroll it in a compute pool. When in doubt, get written permission first.

## 2. Third-party platform Terms of Service — read this before you deploy

**Running MoreGPU workers on free or shared third-party platforms is very often
prohibited by those platforms' Terms of Service.** Many providers explicitly ban
background or idle compute harvesting, cryptocurrency-mining-like workloads, and
long-running non-interactive jobs on their free or trial tiers.

This includes, without limitation:

- **Google Colab** (free and hosted runtimes)
- **Continuous-integration / CI runners**, such as **GitHub Actions**, and other
  free CI/CD systems
- **Free or trial VPS / cloud instances** and "always-free" tiers
- Free tiers of notebook, container, and "playground" hosting services

**Do not run MoreGPU on any such platform unless that platform's Terms of Service
explicitly permit the workload you intend to run** (for example, paid or
self-managed runtimes and instances that you control and that allow background
compute). If the ToS is silent or ambiguous, treat it as prohibited.

MoreGPU ships a **generic worker joiner** that runs on any runtime you are permitted
to use (including a notebook you control, such as `examples/colab_worker.ipynb`). It
provides **no automation for evading** any platform's Terms of Service, quotas, or
bot-detection, and its maintainer will not add any. The absence of a technical
barrier is **not** permission.

Violating a platform's Terms of Service can result in account suspension or
termination, forfeiture of resources, and — depending on jurisdiction and
circumstances — civil or criminal liability. That risk is **entirely yours.**

## 3. Prohibited uses

You must **not** use MoreGPU to:

1. Access, enroll, or run compute on any machine, account, or network **without
   the owner's authorization**, or in excess of the authorization you were given.
2. Circumvent, disable, or evade access controls, quotas, sandboxing, rate
   limits, resource limits, or usage policies of any system or platform.
3. Violate the **Terms of Service, Acceptable Use Policy, or other agreement** of
   any third-party platform, host, employer, school, or network operator
   (including but not limited to the platforms listed in Section 2).
4. Conceal the nature of the workload from a platform or resource owner, or
   misrepresent MoreGPU traffic or compute in order to obtain resources you would
   not otherwise be granted.
5. Perform cryptocurrency mining or comparable resource-harvesting activity on
   any infrastructure where such activity is not expressly permitted.
6. Violate any applicable law or regulation, including computer-misuse,
   unauthorized-access, export-control, sanctions, privacy, and data-protection
   laws.
7. Process, compute over, or transmit data that you are not authorized to
   process, or that would infringe the rights of others (including intellectual
   property, privacy, or confidentiality rights).
8. Attack, overload, or disrupt any system, network, or service (e.g. denial of
   service), or use the pool as a vector for malware, botnet, or command-and-
   control activity.
9. Interfere with the safety, security, or lawful operation of others' systems,
   or endanger health or safety.

The AES-GCM sealing of work units in MoreGPU is a data-in-transit protection
mechanism for pool operators; it is **not** a license or a means to hide
unauthorized or unlawful activity from platform providers or authorities.

## 4. Operator responsibility

If you run an admin server, you are the **operator** of that pool. You are
responsible for:

- keeping your admin token, worker join token, and encryption key confidential
  (each pool generates its own via the first-run wizard; pools are isolated and
  no one shares another pool's tokens);
- ensuring every worker that joins your pool is on an **authorized** machine;
- ensuring your and your participants' use complies with this AUP, all
  applicable third-party Terms of Service, and all applicable law;
- configuring the duty-cycle throttle and other settings responsibly; and
- the compute that your pool performs and the data it processes.

If you contribute a worker to someone else's pool, you remain responsible for
confirming that **your** machine is authorized for this use.

**All responsibility, legal risk, and compliance obligations rest entirely with
the operator and users. The maintainer and contributors are not a party to, and
take no responsibility for, how any pool is deployed or used.**

## 5. No warranty; "AS IS"

MoreGPU is provided **"AS IS", without warranty of any kind**, express or
implied, including but not limited to the warranties of merchantability, fitness
for a particular purpose, title, and non-infringement, as set out in the
Apache-2.0 license. To the maximum extent permitted by law, the maintainer and
contributors **accept no liability** for any claim, damages, loss, or other
liability arising from or in connection with the software or its use, including
any violation of third-party Terms of Service or law by any operator or user.

Use of MoreGPU is entirely at your own risk.

## 6. Enforcement

We may, at our discretion and without obligation, decline support, remove
contributions, or take other reasonable action in response to uses that violate
this AUP. This AUP does not create any right, warranty, or obligation on the
maintainer beyond what the Apache-2.0 license provides.

## 7. Changes

This AUP may be updated from time to time. Material changes will be reflected in
this file in the repository. Your continued use after a change constitutes
acceptance of the updated policy.

---

_If you are unsure whether a particular deployment is permitted, the safe answer
is: don't deploy it until you have confirmed authorization and Terms-of-Service
compliance in writing._

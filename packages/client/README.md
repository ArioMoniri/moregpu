# @moregpu/client

Client SDK for the [MoreGPU](https://github.com/ArioMoniri/moregpu) distributed GPU compute pool.
Submit compute/tensor jobs, send your own data and get results, and read fleet contribution — in
Deno, Node, or the browser.

```ts
import { MoreGPUClient } from '@moregpu/client';
const pool = new MoreGPUClient({ baseUrl: 'http://ADMIN:8787', adminToken: '<admin-token>' });
const C = await pool.matmul([1,2,3, 4,5,6], [7,8, 9,10, 11,12], 2, 2, 3); // → [58,64,139,154]
const dev = await pool.device();  // the pool as a GPU slot
```

Apache-2.0 © Ariorad Moniri.

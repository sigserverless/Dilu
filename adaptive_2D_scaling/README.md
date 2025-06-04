# Adaptive 2D Scaling

## Overview
`adaptive_2D_scaling` offers two complementary scaling mechanisms that correspond to specific sections of our paper:

* **vertical_scaling** — implements the “Dynamic and Fast Scaling Up/Down” subsection.  
* **horizontal_scaling** — implements the “Lazy Scaling Out/In” subsection.



## Architecture

* **Client–Server design**  
  * **Server: Real-time CUDA Kernel Manager (RCKM)**  
    * Distributes execution tokens to each container based on multiple runtime signals.  
  * **Client: CUDA kernel hook logic**  
    * Intercepts driver-level CUDA kernel launches, requests a token from RCKM, then decides whether to release or delay the kernel.

* **Design inspirations**  
  * GaiaGPU <https://github.com/tkestack/vcuda-controller.git>  
  * TGS   <https://github.com/pkusys/TGS>

* **Main extensions**  
  * The token-based scheduling mechanism has been **extended to CUDA 12.X**, whose Runtime API invocation changed fundamentally. Detailed compatibility notes are provided in `client_CUDA12X/`.

## Relationship to cluster_scheduling

The core scheduling code used by `horizontal_scaling` is shared with the `cluster_scheduling` module.  
For implementation details, please refer to `cluster_scheduling/README.md`.
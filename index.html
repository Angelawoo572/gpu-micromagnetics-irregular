<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>GPU Micromagnetic Simulation</title>

  <style>
    body {
      font-family: Arial, Helvetica, sans-serif;
      margin: 40px;
      line-height: 1.55;
      color: #222;
      max-width: 1200px;
    }

    h1 {
      margin-bottom: 0.2em;
    }

    h2 {
      margin-top: 2em;
      border-bottom: 2px solid #ddd;
      padding-bottom: 0.3em;
    }

    h3 {
      margin-bottom: 0.3em;
    }

    a {
      color: #0645ad;
    }

    .summary {
      max-width: 900px;
      font-size: 1.05em;
    }

    .links li {
      margin-bottom: 0.4em;
    }

    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 24px;
      margin-top: 20px;
    }

    .card {
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 16px;
      background: #fafafa;
    }

    .card img {
      width: 100%;
      max-height: 360px;
      object-fit: contain;
      background: white;
      border: 1px solid #ddd;
    }

    .card p {
      margin-bottom: 0;
    }

    .featured {
      border: 2px solid #444;
      background: #f4f4f4;
    }

    embed {
      width: 100%;
      height: 650px;
      border: 1px solid #ccc;
    }

    .note {
      color: #555;
      font-size: 0.95em;
    }
  </style>
</head>

<body>

  <h1>GPU Micromagnetic Simulation with Irregular Geometry</h1>

  <p class="summary">
    This project studies GPU-accelerated micromagnetic simulation using CUDA,
    cuFFT, and SUNDIALS/CVODE. The goal is to simulate 1D, 2D, FFT-based,
    and irregular-geometry Landau-Lifshitz-Gilbert dynamics, with emphasis on
    how geometry, demagnetization fields, and GPU implementation choices affect
    both physical behavior and performance.
  </p>

  <h2>Links</h2>
  <ul class="links">
    <li>
      <a href="https://github.com/Angelawoo572/gpu-micromagnetics-irregular" target="_blank">
        GitHub Repository
      </a>
    </li>
    <li>
      <a href="report/proposal_ode_solver.pdf" target="_blank">
        Project Proposal PDF
      </a>
    </li>
    <li>
      <a href="report/report_ode_solver.pdf" target="_blank">
        Milestone Report PDF
      </a>
    </li>
    <li>
      Final Report PDF: coming soon
    </li>
  </ul>

  <h2>What Physical Problem Is Simulated?</h2>

  <p>
    The project simulates micromagnetic domain evolution in thin magnetic films.
    The examples are motivated by classical magnetic-domain structures such as
    Bloch walls, Néel-like spike domains near defects, cross-tie structures, and
    asymmetric domain walls. In the code, these physical ideas are represented
    by the Landau-Lifshitz-Gilbert equation with exchange, anisotropy, DMI-like
    local interactions, demagnetization fields, and irregular masks such as
    holes, rings, or dead grains.
  </p>

  <p>
    The most advanced version is the irregular FFT-demag simulation: a
    Voronoi-polycrystal magnetic film with dead-grain holes, where local stencil
    physics is combined with a long-range demagnetization field computed by
    cuFFT. This is the closest version to the classical “defect interacting
    with domain walls / spike domains” examples shown in micromagnetics
    literature.
  </p>

  <h2>Highlighted Results</h2>

  <div class="gallery">

    <div class="card featured">
      <h3>Irregular FFT Demag: Voronoi Polycrystal + Dead-Grain Holes</h3>
      <a href="https://github.com/Angelawoo572/gpu-micromagnetics-irregular/blob/main/results/irregular/2d_fft/i3_quiver_animation.gif" target="_blank">
        <img src="results/irregular/2d_fft/i3_quiver_animation.gif" alt="i3 quiver animation">
      </a>
      <p>
        This is the main showcase simulation. It combines irregular grain
        geometry, dead-grain holes, local magnetic interactions, and FFT-based
        demagnetization. It is the strongest example for showing domain-wall
        bending, defect effects, and non-uniform magnetization dynamics.
      </p>
    </div>

    <div class="card featured">
      <h3>i3 Top View: Magnetization Component Map</h3>
      <a href="https://github.com/Angelawoo572/gpu-micromagnetics-irregular/blob/main/results/irregular/2d_fft/i3_mx_topview.gif" target="_blank">
        <img src="results/irregular/2d_fft/i3_mx_topview.gif" alt="i3 mx topview">
      </a>
      <p>
        Top-view visualization of the same irregular FFT-demag system. This
        view makes the domain evolution easier to compare with classical
        microscopy images of spike domains and domain-wall motion near defects.
      </p>
    </div>

    <div class="card">
      <h3>1D Correctness Test</h3>
      <a href="https://github.com/Angelawoo572/gpu-micromagnetics-irregular/blob/main/results/1d/correctness_filed.png" target="_blank">
        <img src="results/1d/correctness_filed.png" alt="1D correctness field">
      </a>
      <p>
        A simpler 1D test case used to check correctness before moving to 2D
        and irregular geometries.
      </p>
    </div>

    <div class="card">
      <h3>2D Periodic Simulation</h3>
      <a href="https://github.com/Angelawoo572/gpu-micromagnetics-irregular/blob/main/results/2d/2d_heter/quiver_animation.gif" target="_blank">
        <img src="results/2d/2d_heter/quiver_animation.gif" alt="2D periodic quiver animation">
      </a>
      <p>
        Baseline 2D periodic simulation. This version shows the transition from
        simple regular-grid dynamics toward more complex 2D magnetic textures.
      </p>
    </div>

    <div class="card">
      <h3>2D FFT Demagnetization</h3>
      <a href="https://github.com/Angelawoo572/gpu-micromagnetics-irregular/blob/main/results/2d_fft/2d_heter.gif" target="_blank">
        <img src="results/2d_fft/2d_heter.gif" alt="2D FFT demag simulation">
      </a>
      <p>
        Extension of the 2D solver with FFT-based long-range demagnetization.
        This adds the nonlocal field needed for realistic thin-film domain
        behavior.
      </p>
    </div>

    <div class="card">
      <h3>Irregular Antidot / Hole Geometry</h3>
      <a href="https://github.com/Angelawoo572/gpu-micromagnetics-irregular/blob/main/results/irregular/2d_fft/2d_i1_768*256_2.gif" target="_blank">
        <img src="results/irregular/2d_fft/2d_i1_768*256_2.gif" alt="irregular hole geometry">
      </a>
      <p>
        Irregular geometry example with FFT demagnetization. This tests how
        holes and masks perturb the magnetization field.
      </p>
    </div>

  </div>

  <h2>Additional Irregular Geometry Examples</h2>

  <div class="gallery">

    <div class="card">
      <h3>i4 Geometry, 1:4 Ratio</h3>
      <a href="https://github.com/Angelawoo572/gpu-micromagnetics-irregular/blob/main/results/irregular/2d_fft/1%3A4_i4_mx_topview.gif" target="_blank">
        <img src="results/irregular/2d_fft/1%3A4_i4_mx_topview.gif" alt="i4 1:4 mx topview">
      </a>
      <p>
        Geometry-ratio sweep showing how changing the shape of the irregular
        region changes the domain pattern.
      </p>
    </div>

    <div class="card">
      <h3>i4 Geometry, 1:8 Ratio</h3>
      <a href="https://github.com/Angelawoo572/gpu-micromagnetics-irregular/blob/main/results/irregular/2d_fft/1%3A8_i4_mx_topview.gif" target="_blank">
        <img src="results/irregular/2d_fft/1%3A8_i4_mx_topview.gif" alt="i4 1:8 mx topview">
      </a>
      <p>
        A more elongated geometry case, useful for comparing how aspect ratio
        affects domain-wall motion and field concentration.
      </p>
    </div>

    <div class="card">
      <h3>i5 Ring / Ring-Hole Geometry</h3>
      <a href="https://github.com/Angelawoo572/gpu-micromagnetics-irregular/blob/main/results/irregular/2d_fft/i5_mx_topview.gif" target="_blank">
        <img src="results/irregular/2d_fft/i5_mx_topview.gif" alt="i5 mx topview">
      </a>
      <p>
        Ring-based irregular geometry. This tests whether the same solver can
        support different active and inactive mask layouts with only small code
        changes.
      </p>
    </div>

    <div class="card">
      <h3>i6 Circular Geometry</h3>
      <a href="https://github.com/Angelawoo572/gpu-micromagnetics-irregular/blob/main/results/irregular/2d_fft/i6_circlemx_topview.gif" target="_blank">
        <img src="results/irregular/2d_fft/i6_circlemx_topview.gif" alt="i6 circle mx topview">
      </a>
      <p>
        Circular-mask example showing another irregular geometry case. This is
        useful for demonstrating that the implementation is not restricted to a
        single hand-coded shape.
      </p>
    </div>

  </div>

  <h2>Code Organization</h2>

  <p>
    The source code is organized under <code>src/</code>. Input and configuration
    files are placed under <code>configs/</code>. Generated figures, animations,
    and profiling outputs are stored under <code>results/</code>. The
    <code>report/</code> folder contains the proposal, milestone report, and
    final report when available.
  </p>

  <h2>Proposal Preview</h2>
  <embed src="report/proposal_ode_solver.pdf" type="application/pdf">

  <h2>Milestone Report Preview</h2>
  <embed src="report/report_ode_solver.pdf" type="application/pdf">

</body>
</html>
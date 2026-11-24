# Scientific Basis of Organic Photovoltaics (OPV)

## 1. Operating Mechanism of Bulk Heterojunction (BHJ) Solar Cells

Organic photovoltaic cells convert incident photon energy into electricity through several sequential photophysical steps:

1. **Photon Absorption & Exciton Generation**: Light absorption in the donor or acceptor material generates a strongly Coulomb-bound electron–hole pair (Frenkel exciton) due to the low dielectric constant ($\epsilon_r \approx 3-4$) of organic semiconductors.
2. **Exciton Diffusion**: The exciton diffuses through the organic phase toward a donor–acceptor heterojunction interface. Typical exciton diffusion lengths in organic semiconductors are on the order of $10\text{ nm}$.
3. **Charge Transfer & Exciton Dissociation**: At the donor–acceptor interface, an energetic offset (between donor LUMO and acceptor LUMO, or donor HOMO and acceptor HOMO) provides the driving force to dissociate the exciton into a charge transfer (CT) state and free charge carriers (electrons in the acceptor, holes in the donor).
4. **Charge Transport & Collection**: Free electrons and holes drift under the internal electric field toward their respective selective contacts (cathode and anode).

## 2. Four Key Device Performance Metrics

The overall Power Conversion Efficiency ($PCE$) is defined as:

$$PCE = \frac{P_{max}}{P_{in}} = \frac{V_{OC} \times J_{SC} \times FF}{P_{in}}$$

where:
- **$P_{in}$**: Incident solar irradiance (standard AM1.5G reference spectrum is $100\text{ mW/cm}^2$ or $1000\text{ W/m}^2$).
- **$V_{OC}$ (Open-Circuit Voltage, V)**: The voltage across the solar cell when the net current is zero. It is governed by the effective bandgap between the donor HOMO and acceptor LUMO, reduced by energetic disorder and radiative/non-radiative recombination losses:
  $$V_{OC} \approx \frac{1}{q} (E_{LUMO}^{Acceptor} - E_{HOMO}^{Donor}) - \Delta V_{loss}$$
- **$J_{SC}$ (Short-Circuit Current Density, mA/cm²)**: The current density generated when voltage equals zero. It depends on optical absorption bandwidth, active layer thickness, exciton dissociation yield, and charge carrier mobility.
- **$FF$ (Fill Factor, % or ratio)**: The ratio of maximum deliverable electrical power to the product $V_{OC} \times J_{SC}$. It reflects internal series and shunt resistances, charge recombination rates, and balanced carrier mobility.

## 3. Clear Distinctions in Data Modeling

In OPV-PredictNet, the system strictly separates four distinct concepts:

1. **Molecular Candidate**:
   - A single chemical species (either donor or acceptor).
   - Characterized by molecular formula, molecular weight, topological polar surface area (TPSA), conjugated backbone topology, and computed/gas-phase HOMO/LUMO energies.
   - **Crucial Rule**: An isolated molecule alone does not have a "Power Conversion Efficiency".

2. **Donor–Acceptor Material Combination**:
   - A blend pair (e.g., PM6:Y6, PTB7-Th:PC71BM).
   - Characterized by frontier orbital alignments ($\Delta E_{LUMO} = E_{LUMO}^{Donor} - E_{LUMO}^{Acceptor}$, $\Delta E_{HOMO}$), complementary optical absorption spectra, miscibility, and blend ratio (e.g., 1:1.2).

3. **Experimental OPV Device Record**:
   - A physically fabricated solar cell.
   - Includes full stack architecture (e.g., ITO / PEDOT:PSS / Active Layer / PFN-Br / Al), casting solvent (e.g., chloroform, chlorobenzene), processing additives (e.g., 1,8-diiodooctane / DIO, 1-chloronaphthalene / CN), thermal annealing conditions, film thickness, active device area, and ambient test atmosphere.

4. **Predicted Device Performance Value**:
   - A statistical, data-driven estimate of device metrics ($PCE$, $V_{OC}$, $J_{SC}$, $FF$) output by machine learning models trained on published empirical device distributions.
   - Inherently accompanied by uncertainty bounds and governed by model domain of applicability.

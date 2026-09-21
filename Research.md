Something goes into the water at a chosen point in R: a container off a ship, a life raft, a slick of oil. Where has the surface flow taken it a week later? A month later? A year? At the scale of the whole box, which parts of R feed which others, and which are cut off from the rest?

Product. A surface-transport operator for R. E.g. a transition matrix (or similar) someone else could pick up and iterate forward, together with what it says about where a few chosen release points end up, and about where R as a whole gathers material and where it loses it.


Regions:

---
Agulhas region:
What to track?
Shipping container:
- Would be easiest to track. The wind has almost no effect on the movement of the container (windage), and hence the current will be the only impact. 

Oil Spill:
-  Oil at the surface drifts at roughly current + 0.035 × wind. You'll need ERA5 10 m wind fields alongside your current data.
- ERA5 is freely available via the Copernicus Climate Data Store (cds.climate.copernicus.eu)
- assign each particle a mass that decreases over time (exponential decay approximating evaporation + emulsification). This lets you answer "how much surface oil remains after X days?" as well as "where is it?"



Final Output:
- Transition matrix iterating through time to show where the output ends up at specific times (e.g 1 week, 1 month, 1 year)
# Source-Grounded Benchmark Summary

## Public Sources
- TravelPlanner README, website, metadata, and sample rows
- OpenFlights airports.dat and routes.dat
- OpenStreetMap/Overpass city snapshots for hotels and attractions

## Normalized Outputs
- cities.json: 8 rows
- flights.csv: 288 rows
- hotels.csv: 96 rows
- attractions.csv: 96 rows
- routes.csv: 672 rows

## Derived Fields
- Flight dates, times, fares, seat counts, baggage, and refund policy are derived deterministically from OpenFlights route structure.
- Hotel prices, availability, breakfast, and refund policy are derived deterministically from OSM metadata and city cost factors.
- Attraction categories and ticket prices are derived deterministically from OSM tags and name heuristics.
- Attack scenarios are created locally for controlled evaluation.

# smurf_detector
Community project to detect smurfs accounts in AoEIIDE based on publicly available information.

## What is in

Scraped aoe2insights for match stats that every profile has, i.e. winrates for various game lengths. 
It is used as main indicator for potential smurf classification but it is not enough.
I also implemented download of match database from aoestats.io that can be used for detailed match investigation.

## New features

### Match inspection
Many people in Reddit thread suggested to distinguish between map pickers/quitters, people loosing initial games intentionally and actual smurfs.
The only way to do it is inspect games played and dropped. But there are many games played so winrates can be used as coarse filter and detailed match investigation as next step.
Did not have time to look into it.

### Alt accounts

Also, it would be nice to link potential smurf with their main account. I've seen page that tracks that but cannot find it now.

### Chat

Some smurfs talk crap apparently. It could maybe be used?

## Requried packages
- pandas
- requests
- pickle
- matplotlib
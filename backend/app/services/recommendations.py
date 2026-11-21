RECOMMENDATIONS = {
    "plastic": "Empty and rinse it. Put recyclable plastic in the plastics recycling stream where accepted; keep contaminated film or mixed plastic out of recycling.",
    "paper": "Keep it clean and dry, then place it in paper recycling. Food-soiled paper should go in trash unless your local program accepts composting.",
    "cardboard": "Flatten clean cardboard boxes and place them in cardboard or mixed-paper recycling. Remove plastic wrap and food residue.",
    "glass": "Rinse glass containers and use the local glass-recycling bin if available. Never place broken glass loose in recycling; wrap it safely for disposal.",
    "metal": "Empty and rinse cans, then recycle them with metals. Put sharp lids inside the can and crimp the opening where practical.",
    "trash": "This item is not identified as a standard recyclable. Bag it securely and use the general-waste bin; check local guidance for special materials.",
}

def recommendation_for(label: str) -> str:
    return RECOMMENDATIONS.get(label.lower(), "Check your local waste authority for the correct disposal route.")

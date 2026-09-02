"""Prompt set for the entropy/watermark investigation (separate from the demo's 12 curated
prompts). 80 prompts across 4 categories, 20 each, to test -- with a real sample size rather
than one anecdotal example per side -- whether numeric/unit continuations are reliably
lower-entropy (and thus more watermark-resistant) than procedural, equipment-naming, or
generic language.

Category is encoded as a prefix in each prompt_id (NUM/VERB/EQUIP/GEN) so downstream analysis
can group by category without any change to the existing extraction code.
"""

NUMERIC_UNIT_PROMPTS = [
    ("NUM01", "Pressure test the tubing to 5,000 psi for 15"),
    ("NUM02", "The BOP was tested to 10,000 psi for 10"),
    ("NUM03", "Circulate bottoms up at 650 gpm for 2"),
    ("NUM04", "The kill line was pressure tested to 3,000 psi for 5"),
    ("NUM05", "Run in hole to a depth of 12,450"),
    ("NUM06", "The cement was allowed to cure for 12"),
    ("NUM07", "Displace the well with 850 barrels of"),
    ("NUM08", "The choke was set at 64/64"),
    ("NUM09", "Torque the connection to 18,500 ft-"),
    ("NUM10", "The mud weight was increased to 12.5 ppg from"),
    ("NUM11", "POOH from a depth of 9,200"),
    ("NUM12", "The annulus pressure stabilized at 250"),
    ("NUM13", "Pump the spacer at a rate of 4 barrels per"),
    ("NUM14", "The liner hanger was set at 8,750"),
    ("NUM15", "Hold the pressure test for a minimum of 30"),
    ("NUM16", "The perforating guns were run to 11,300"),
    ("NUM17", "Increase pump rate to 6 barrels per"),
    ("NUM18", "The wellhead pressure read 1,450"),
    ("NUM19", "Squeeze cement was pumped at 2 barrels per"),
    ("NUM20", "The casing was landed at 13,200"),
]

VERB_PROCEDURAL_PROMPTS = [
    ("VERB01", "The crew circulated the well clean and prepared to"),
    ("VERB02", "After the packer was set, the operator continued to"),
    ("VERB03", "With TD reached, the drilling team began to"),
    ("VERB04", "The liner was run to depth, and the crew immediately"),
    ("VERB05", "Once cement was pumped and displaced, the team continued to"),
    ("VERB06", "The pressure test held, so the crew prepared to"),
    ("VERB07", "The wellhead was nippled up, and the operator started to"),
    ("VERB08", "After pulling the BHA to surface, the crew began to"),
    ("VERB09", "The BOP was function tested, and the crew then"),
    ("VERB10", "The pill was spotted across the zone, and the operator prepared to"),
    ("VERB11", "The fishing tool reached the top of fish and began to"),
    ("VERB12", "Once the wiper trip was completed, the crew started to"),
    ("VERB13", "The perforating guns were rigged up, and the crew prepared to"),
    ("VERB14", "After the coiled tubing unit was rigged up, the operator began to"),
    ("VERB15", "The stimulation treatment was pumped, and the crew immediately"),
    ("VERB16", "Once the packer was retrieved, the operator continued to"),
    ("VERB17", "The completion string was run in hole, and the crew started to"),
    ("VERB18", "After the wellhead was installed, the operator began to"),
    ("VERB19", "The slickline unit was rigged up, and the crew prepared to"),
    ("VERB20", "Once the tubing was landed, the operator continued to"),
]

EQUIPMENT_ENTITY_PROMPTS = [
    ("EQUIP01", "The crew rigged up the"),
    ("EQUIP02", "The operator installed a new"),
    ("EQUIP03", "A fishing job was performed using a"),
    ("EQUIP04", "The completion included a"),
    ("EQUIP05", "The BHA consisted of a bit, a"),
    ("EQUIP06", "The wellhead was equipped with a"),
    ("EQUIP07", "The string included a packer and a"),
    ("EQUIP08", "The rig floor crew connected the"),
    ("EQUIP09", "The tool string was made up with a"),
    ("EQUIP10", "The workover rig was equipped with a"),
    ("EQUIP11", "A new section of casing was run using a"),
    ("EQUIP12", "The intervention was carried out with a"),
    ("EQUIP13", "The perforating string included a"),
    ("EQUIP14", "The stimulation used a"),
    ("EQUIP15", "The lower completion consisted of a"),
    ("EQUIP16", "The tubing string was landed with a"),
    ("EQUIP17", "A gauge was installed to monitor the"),
    ("EQUIP18", "The safety valve was tested using a"),
    ("EQUIP19", "The subsea tree included a"),
    ("EQUIP20", "The workstring was tripped in with a"),
]

GENERIC_CONTROL_PROMPTS = [
    ("GEN01", "It was a pleasant afternoon, and the children decided to"),
    ("GEN02", "The chef prepared a meal and then began to"),
    ("GEN03", "After finishing the book, she decided to"),
    ("GEN04", "The team won the match and started to"),
    ("GEN05", "He looked out the window and began to"),
    ("GEN06", "The garden was blooming, so she decided to"),
    ("GEN07", "After the meeting ended, everyone started to"),
    ("GEN08", "The weather was nice, so they decided to"),
    ("GEN09", "She opened the door and began to"),
    ("GEN10", "The concert finished, and the crowd started to"),
    ("GEN11", "He finished his coffee and decided to"),
    ("GEN12", "The movie ended, and they began to"),
    ("GEN13", "After the rain stopped, the kids started to"),
    ("GEN14", "She packed her bags and prepared to"),
    ("GEN15", "The store closed, and the staff began to"),
    ("GEN16", "He read the letter and decided to"),
    ("GEN17", "The sun set, and they began to"),
    ("GEN18", "After dinner, the family decided to"),
    ("GEN19", "The train arrived, and passengers began to"),
    ("GEN20", "She finished her run and started to"),
]

ALL_PROMPTS = (
    NUMERIC_UNIT_PROMPTS + VERB_PROCEDURAL_PROMPTS
    + EQUIPMENT_ENTITY_PROMPTS + GENERIC_CONTROL_PROMPTS
)


def category_of(prompt_id):
    """Category label from a prompt_id prefix, e.g. 'NUM07' -> 'numeric_unit'."""
    prefix_map = {
        "NUM": "numeric_unit",
        "VERB": "verb_procedural",
        "EQUIP": "equipment_entity",
        "GEN": "generic_control",
    }
    for prefix, label in prefix_map.items():
        if prompt_id.startswith(prefix):
            return label
    raise ValueError(f"Unrecognized prompt_id prefix: {prompt_id!r}")

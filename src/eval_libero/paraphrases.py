"""Recommended language paraphrases for libero_spatial's 10 instructions.

Same object (black bowl), same spatial anchor, same target (plate) as the
original training instruction in every case - only wording/sentence structure
changes (verb swaps: pick up/grab/lift/take, place/put/set; clause
restructuring: prepositional phrase -> relative clause / participle phrase).
Used by both the interactive probe (recovery_probe_gui.py) and the batch
closed-loop benchmark (rollout_sim_paraphrase.py) so the two stay consistent.
"""

PARAPHRASES = {
    0: ["grab the black bowl that sits between the plate and the ramekin, then set it on the plate",
        "take the black bowl located between the ramekin and the plate and place it onto the plate",
        "retrieve the black bowl positioned between the plate and the ramekin, and put it on the plate"],
    1: ["grab the black bowl beside the ramekin and set it down on the plate",
        "lift the black bowl that is next to the ramekin, then put it on the plate",
        "take the black bowl positioned next to the ramekin and place it onto the plate"],
    2: ["take the black bowl sitting in the middle of the table and place it on the plate",
        "grab the black bowl at the center of the table and put it onto the plate",
        "lift the black bowl from the middle of the table and set it on the plate"],
    3: ["lift the black bowl resting on top of the cookie box and set it on the plate",
        "grab the black bowl that is on the cookie box, then place it on the plate",
        "take the black bowl sitting on top of the cookie box and put it onto the plate"],
    4: ["take the black bowl out of the top drawer of the wooden cabinet and put it on the plate",
        "reach into the top drawer of the wooden cabinet, grab the black bowl, and set it on the plate",
        "retrieve the black bowl from the top drawer of the wooden cabinet and place it onto the plate"],
    5: ["lift the black bowl that is on top of the ramekin and place it on the plate",
        "grab the black bowl sitting on the ramekin, then put it onto the plate",
        "take the black bowl resting on the ramekin and set it on the plate"],
    6: ["grab the black bowl beside the cookie box and set it on the plate",
        "take the black bowl next to the cookie box, then place it onto the plate",
        "lift the black bowl positioned next to the cookie box and put it on the plate"],
    7: ["lift the black bowl resting on the stove and put it on the plate",
        "grab the black bowl that is on top of the stove, then set it on the plate",
        "take the black bowl sitting on the stove and place it onto the plate"],
    8: ["grab the black bowl beside the plate and put it onto the plate",
        "take the black bowl next to the plate, then place it on the plate",
        "lift the black bowl positioned beside the plate and set it on the plate"],
    9: ["lift the black bowl resting on top of the wooden cabinet and set it on the plate",
        "grab the black bowl that is on the wooden cabinet, then place it onto the plate",
        "take the black bowl sitting on the wooden cabinet and put it onto the plate"],
}

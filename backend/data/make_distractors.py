"""Generate synthetic distractor documents: same vocabulary/structure as the gold docs but different
entities and numbers. Asserts that no gold fact string leaks into a distractor."""
import json, pathlib, itertools
here = pathlib.Path(__file__).parent
facts = [f.lower() for q in json.loads((here / "eval/gold_qa.json").read_text()) for f in q["facts"]]
H = "[SYNTHETIC SAMPLE - illustrative only, not real regulatory text]\n"
cos = ["Helix Textiles Ltd", "Vega Steel Pvt Ltd", "Nimbus Logistics Ltd", "Aster Foods Pvt Ltd", "Corvus Energy Ltd",
       "Lumen Retail Ltd", "Borealis Motors Ltd", "Sable Chemicals Pvt Ltd"]
docs = {}
# percentages that cannot contain a gold fact as a substring ("12 percent" contains "2 percent")
SAFE = [v for v in range(3, 70) if not str(v).endswith("2") and v not in (15, 18)]
for i, (a, b) in enumerate(zip(cos, cos[1:] + cos[:1])):
    n = 3 + i  # varies numbers
    docs[f"tax_distractor_{i}.txt"] = H + f"""CBIC Circular No. {300+i}/2025-GST - Place of Supply and Credit Notes
Rule 1. ISD Registration. {a} must obtain an Input Service Distributor registration for common credit shared with {b} under Rule 4{i} of the CGST Rules.
Rule 2. E-Invoicing Threshold. E-invoicing applies to taxpayers whose aggregate turnover exceeds INR {20+i*5} crore, as clarified by the GST Council.
Rule 3. Interest. The Tax Authority may levy interest at {SAFE[i]} percent per annum on wrongly availed credit.
Rule 4. Penalty. Delay in filing the ISD return attracts a penalty of INR {10000+i*1500} under Section 12{i} of the CGST Act.
"""
    docs[f"tax_cbdt_distractor_{i}.txt"] = H + f"""CBDT Notification No. {40+i}/2025 - Transfer Pricing Documentation
Section 1. Applicability. {a} and its associated enterprises with international transactions above INR {105+i*10} crore.
Section 2. Safe Harbour Margin. The Income Tax Department prescribes an operating profit margin of {SAFE[i+8]} percent on cost for software development services.
Section 3. Prospective Amendment. The revised threshold is a prospective amendment effective from 1 April 202{7+i%2}.
Section 4. Penalty. Failure to keep documentation attracts a penalty of {SAFE[i+3]} percent under Section 271{chr(65+i)} of the Income Tax Act.
"""
    docs[f"grc_distractor_{i}.txt"] = H + f"""RBI Circular - KYC Periodic Updation and Reporting for {a}
Para 1. Customer Due Diligence. Regulated Entities must complete CDD before onboarding customers, including checks of beneficial owners holding more than {SAFE[i+16]} percent.
Para 2. Reporting. A Suspicious Transaction Report must be furnished to the Financial Intelligence Unit within {3+i} working days of detection.
Para 3. Penalty. The Reserve Bank may impose a penalty of up to INR {21+i} crore for contravention, and the Principal Officer of {a} is responsible.
Para 4. Control Mapping. Internal audit control IA-0{i%6+1} maps to Para 1 and control IA-2{i} maps to Para 2.
"""
    docs[f"deals_distractor_{i}.txt"] = H + f"""Share Purchase Agreement - Acquisition of {a} by {b}
Clause 1. Parties. {b} (the Buyer) acquires {SAFE[i+24]} percent of {a} from its promoters (the Sellers).
Clause 2. Purchase Price. Enterprise value of INR {300+i*45} crore subject to valuation under the Companies Act 2013.
Clause 3. Indemnity. Sellers indemnify the Buyer for pre-closing liabilities, capped at INR {30+i*8} crore with a basket of INR {11+i} crore.
Clause 4. Material Adverse Change. A MAC means any event reducing revenue by more than {SAFE[i+4]} percent; the Buyer may rescind the agreement before closing.
Clause 5. FEMA Compliance. Foreign investment requires Form FC-TRS within {45+i} days.
"""
for name, text in docs.items():
    low = text.lower()
    assert not [f for f in facts if f in low], (name, [f for f in facts if f in low])
    (here / "sample" / name).write_text(text)
print(len(docs), "distractors written")

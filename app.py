"""
AI Event Master Planner (India) - Prototype
-------------------------------------------
A single-file Streamlit app that collects event details, applies a fixed
Indian-market budget split, and asks Gemini (gemini-2.5-flash) to produce a
complete "Master Plan".

Setup (Replit):
    1. pip install streamlit google-genai   (or add them to pyproject/requirements)
    2. Add a Replit Secret named GEMINI_API_KEY (get a key at https://aistudio.google.com/apikey)
    3. Run:  streamlit run app.py --server.port 8080 --server.address 0.0.0.0

NOTE: The "Pay ₹99" button is a DEMO. No real payment is processed.
To go live, integrate Razorpay / Cashfree / UPI before calling Gemini.
"""

import os

import streamlit as st
from google import genai

# ----------------------------------------------------------------------------
# 1. PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Event Master Planner (India)",
    page_icon="🎉",
    layout="centered",
)

# ----------------------------------------------------------------------------
# 2. CONSTANTS
# ----------------------------------------------------------------------------
MODEL_NAME = "gemini-2.5-flash"

# Fixed budget split (percentages must total 100)
BUDGET_SPLIT = {
    "Catering": 30,
    "Venue": 25,
    "Decor": 12,
    "Planner Fee": 10,
    "AV/Sound": 6,
    "Photo/Video": 6,
    "Logistics": 7,
    "Buffer": 4,
}
assert sum(BUDGET_SPLIT.values()) == 100, "Budget split must total 100%"

# Indian standard catering plate range (per head, before GST) and GST rates
PLATE_MIN, PLATE_MAX = 700, 1500
CATERING_GST = 0.05
VENUE_GST = 0.18

CITIES = ["Bengaluru", "Mumbai", "Delhi NCR", "Chennai", "Hyderabad", "Other"]
CATERING_OPTIONS = ["Pure Veg", "Veg & Non-Veg", "Jain Options Included"]
THEMES = ["Traditional", "Modern & Minimal", "Corporate", "I'm not sure", "Other"]


# ----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ----------------------------------------------------------------------------
def inr(amount: float) -> str:
    """Format a number in the Indian digit-grouping style, e.g. 1234567 -> ₹12,34,567."""
    n = int(round(amount))
    s = str(abs(n))
    if len(s) > 3:
        last3, rest = s[-3:], s[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        s = ",".join(groups + [last3])
    return f"{'-' if n < 0 else ''}₹{s}"


def get_gemini_client():
    """Create a Gemini client using the API key from Replit Secrets / env / Streamlit secrets."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def compute_budget(total_budget: float) -> dict:
    """Return the rupee amount for each budget category."""
    return {name: total_budget * pct / 100 for name, pct in BUDGET_SPLIT.items()}


def catering_reality_check(catering_budget: float, guests: int) -> dict:
    """
    Check whether the catering allocation (which we treat as GST-inclusive)
    can cover Indian standard plates of ₹700-₹1500 + 5% GST for all guests.
    """
    min_needed = guests * PLATE_MIN * (1 + CATERING_GST)
    max_needed = guests * PLATE_MAX * (1 + CATERING_GST)
    per_plate_all_in = catering_budget / guests
    per_plate_pre_gst = per_plate_all_in / (1 + CATERING_GST)

    if catering_budget < min_needed:
        status = "unrealistic"
    elif catering_budget < (min_needed + max_needed) / 2:
        status = "tight-but-feasible"
    else:
        status = "comfortable"

    return {
        "min_needed": min_needed,
        "max_needed": max_needed,
        "per_plate_all_in": per_plate_all_in,
        "per_plate_pre_gst": per_plate_pre_gst,
        "status": status,
    }


def build_prompt(inputs: dict, amounts: dict, check: dict) -> str:
    """Construct the detailed master prompt for Gemini, including strict rules."""
    split_lines = "\n".join(
        f"- {name}: {BUDGET_SPLIT[name]}% = {inr(amounts[name])}" for name in BUDGET_SPLIT
    )
    venue_base = amounts["Venue"] / (1 + VENUE_GST)

    return f"""
You are a senior Indian event planner with 15+ years of experience across metro cities.
Create a complete, practical "Event Master Plan" for the event below.

=== EVENT DETAILS ===
- Event Name: {inputs['name']}
- City: {inputs['city']}
- Number of Guests: {inputs['guests']}
- Total Budget: {inr(inputs['budget'])} (INR)
- Catering Preference: {inputs['catering']}
- Theme: {inputs['theme']}

=== STRICT RULES (follow every one) ===

RULE 1 - BUDGET SPLIT (all amounts in INR, use the Indian format such as ₹1,50,000):
Use EXACTLY this split. These figures are already calculated; do not change them.
{split_lines}
Present this as a markdown table with columns: Category | % | Amount (INR) | What it covers.

RULE 2 - CATERING REALITY CHECK:
- Indian standard catering plates cost ₹{PLATE_MIN} to ₹{PLATE_MAX} per head plus {int(CATERING_GST * 100)}% GST.
- Catering allocation: {inr(amounts['Catering'])} for {inputs['guests']} guests
  = about {inr(check['per_plate_all_in'])} per guest including GST
  (about {inr(check['per_plate_pre_gst'])} before GST).
- Required for this guest count: {inr(check['min_needed'])} (basic) to {inr(check['max_needed'])} (premium), GST included.
- Pre-computed verdict: {check['status'].upper()}.
Write a clear "Reality Check" section confirming or disputing this verdict. If the verdict is
"unrealistic", open the section with a bold WARNING and give concrete options (reduce guests,
simplify the menu, raise the budget, or shift money from other categories). If it is feasible,
say what quality of menu this per-plate amount can realistically buy.
Also suggest a sample menu that suits "{inputs['catering']}" (strictly respect this preference;
for Jain options, avoid onion, garlic and root vegetables in the Jain counter).

RULE 3 - VENUES:
Suggest exactly 3 realistic VENUE TYPES available in {inputs['city']} (for example banquet hall,
farmhouse, hotel ballroom, rooftop, convention centre, community hall, resort lawn).
The venue budget is {inr(amounts['Venue'])} INCLUDING {int(VENUE_GST * 100)}% GST
(about {inr(venue_base)} before GST). For each venue type give: why it fits the theme,
typical capacity fit for {inputs['guests']} guests, an estimated price range inside this budget,
the GST shown separately, and one pro tip. Mention typical popular localities in {inputs['city']}
only in general terms. Do NOT invent specific business names or phone numbers.

RULE 4 - PERMITS & LEGAL COMPLIANCE (India):
List the mandatory permits with who issues them, rough timeline, and a one-line how-to:
- PPL and IPRS music licences (for playing recorded and live music publicly)
- Police NOC / permission from the local police station or licensing authority
- FSSAI compliance (licence or registration for caterers; verify the caterer's FSSAI number)
- Outdoor sound curfew: loudspeakers and music must stop by 10 PM for outdoor events
  (state clearly that this is a hard deadline and that the plan's schedule respects it)
Add a short note that exact rules and fees vary by state and city, and that the organiser must
confirm with the local authorities. Add any other commonly required permit for {inputs['city']}
(for example fire safety or municipal permission) only if relevant.

RULE 5 - PROCUREMENT LIST (heavy equipment):
Give a table of heavy equipment to procure or rent, with suggested quantity for
{inputs['guests']} guests, and an estimated rental range in INR, covering at least:
- Diesel gensets (with capacity in kVA and a backup unit)
- ACs / air coolers / misting fans (as suited to the city's climate and the venue type)
- Bouncers / security staff (suggested count)
- Stage trussing (and lighting rigs)
Mention which budget category each item falls under (AV/Sound, Logistics or Decor) so the
totals stay within the split above.

=== ADDITIONAL SECTIONS ===
Also include these, kept concise:
1. Executive Summary (3-4 lines).
2. Theme & Decor Ideas tailored to "{inputs['theme']}" (if the theme is "I'm not sure", recommend
   the best theme for this event, budget and city, and explain why).
3. Photo/Video & AV/Sound recommendations within their budget lines.
4. A planning timeline (countdown checklist from 8 weeks before to event day).
5. Top 5 risks and how to avoid them (weather, power cuts, vendor no-shows, etc.).

=== STYLE ===
- Use clean markdown with headings (##), tables, and bullet points.
- Keep every figure in INR using Indian digit grouping.
- Be honest and realistic; do not inflate what the budget can buy.
- End with a one-line reminder that all prices are estimates and should be confirmed with vendors.
""".strip()


def generate_master_plan(prompt: str) -> str:
    """Call Gemini and return the generated plan text."""
    client = get_gemini_client()
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Add it in Replit: Tools > Secrets, then restart the app."
        )
    response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
    if not response.text:
        raise RuntimeError("Gemini returned an empty response. Please try again.")
    return response.text


# ----------------------------------------------------------------------------
# 4. UI - HEADER
# ----------------------------------------------------------------------------
st.title("🎉 AI Event Master Planner")
st.caption("Plan any Indian event in minutes: budget split, venues, permits and procurement.")

# Session state keeps the generated plan visible across Streamlit reruns
st.session_state.setdefault("plan", None)
st.session_state.setdefault("plan_meta", None)

# ----------------------------------------------------------------------------
# 5. UI - QUESTIONNAIRE
# (No st.form here, so the "Other" theme box can appear instantly.)
# ----------------------------------------------------------------------------
st.subheader("📝 Tell us about your event")

event_name = st.text_input("Event Name", placeholder="e.g. Sharma Wedding Reception")

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("City", CITIES)
    guests = st.number_input(
        "Number of Guests", min_value=10, max_value=10000, value=150, step=10
    )
    catering = st.selectbox("Catering Preference", CATERING_OPTIONS)
with col2:
    budget = st.number_input(
        "Total Budget in ₹", min_value=25000, value=500000, step=5000
    )
    theme_choice = st.selectbox("Theme", THEMES)

# Conditional logic: show a custom theme box only when "Other" is selected
custom_theme = ""
if theme_choice == "Other":
    custom_theme = st.text_input(
        "Describe your custom theme", placeholder="e.g. Rajasthani Royal Darbar"
    )

final_theme = custom_theme.strip() if theme_choice == "Other" else theme_choice

# Live budget preview (calculated locally, no API cost)
with st.expander("💰 Live budget preview", expanded=False):
    preview = compute_budget(budget)
    for cat, amt in preview.items():
        st.write(f"**{cat}** ({BUDGET_SPLIT[cat]}%): {inr(amt)}")

st.divider()

# ----------------------------------------------------------------------------
# 6. UI - MONETIZATION BUTTON + AI LOGIC
# ----------------------------------------------------------------------------
st.info("🔒 Demo mode: no real payment is taken in this prototype.")

if st.button("Pay ₹99 & Generate Master Plan", type="primary", use_container_width=True):
    # --- Basic input validation ---
    if not event_name.strip():
        st.error("Please enter an Event Name.")
    elif theme_choice == "Other" and not final_theme:
        st.error("Please type your custom theme.")
    else:
        # --- Pre-calculate numbers in Python so the AI uses accurate figures ---
        amounts = compute_budget(budget)
        check = catering_reality_check(amounts["Catering"], int(guests))

        inputs = {
            "name": event_name.strip(),
            "city": city,
            "guests": int(guests),
            "budget": budget,
            "catering": catering,
            "theme": final_theme,
        }

        # --- Build the master prompt (f-string inside build_prompt) ---
        prompt = build_prompt(inputs, amounts, check)

        # --- Call Gemini with a loading spinner ---
        try:
            with st.spinner("Our AI planner is crafting your Master Plan... (about 20-40 seconds)"):
                plan_text = generate_master_plan(prompt)
            st.session_state["plan"] = plan_text
            st.session_state["plan_meta"] = {"inputs": inputs, "check": check, "amounts": amounts}
        except Exception as e:
            st.session_state["plan"] = None
            st.error(f"Could not generate the plan: {e}")

# ----------------------------------------------------------------------------
# 7. UI - OUTPUT
# ----------------------------------------------------------------------------
if st.session_state["plan"]:
    meta = st.session_state["plan_meta"]
    chk = meta["check"]
    inp = meta["inputs"]

    st.success("✅ Your Master Plan is ready!")

    # Instant, locally computed catering reality-check banner
    if chk["status"] == "unrealistic":
        st.warning(
            f"⚠️ Budget alert: {inr(meta['amounts']['Catering'])} for catering works out to about "
            f"{inr(chk['per_plate_all_in'])} per guest (GST included). You need at least "
            f"{inr(chk['min_needed'])} for {inp['guests']} guests at standard plate rates."
        )
    else:
        st.caption(
            f"Catering check: about {inr(chk['per_plate_all_in'])} per guest incl. GST "
            f"({chk['status'].replace('-', ' ')})."
        )

    st.markdown(st.session_state["plan"])

    st.download_button(
        "⬇️ Download Plan (Markdown)",
        data=st.session_state["plan"],
        file_name=f"{inp['name'].replace(' ', '_')}_master_plan.md",
        mime="text/markdown",
    )

    # ------------------------------------------------------------------------
    # 8. SPONSORED VENDOR AD (dummy, demonstrates the B2B revenue model)
    # ------------------------------------------------------------------------
    st.divider()
    st.markdown(
        f"""
<div style="border:2px dashed #FF9933; border-radius:12px; padding:18px; background:#FFF8F0; color:#222;">
  <span style="font-size:12px; color:#888; letter-spacing:1px;">SPONSORED VENDOR</span>
  <h3 style="margin:6px 0; color:#B45F06;">🏛️ Royal Orchid Banquets</h3>
  <p style="margin:0 0 8px 0;">
    Get <b>15% OFF</b> venue bookings in {inp['city'] if inp['city'] != 'Other' else 'your city'}
    when you book through Master Planner.
  </p>
  <p style="margin:0; font-size:13px;">Use code <b>MASTER15</b> &middot; Valid for the next 30 days</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption("Sample ad for demonstration. Vendors pay per lead or a listing fee (B2B revenue).")

# ----------------------------------------------------------------------------
# 9. FOOTER
# ----------------------------------------------------------------------------
st.divider()
st.caption(
    "AI-generated estimates only. Always confirm prices, permits and regulations "
    "with vendors and local authorities."
)

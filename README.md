    ap.add_argument("--query", "-q",
                    help="user prompt to drive activity selection; "
                         "if omitted, asks interactively")
    ap.add_argument("--random", action="store_true",
                    help="ignore --query and pick activities at random")
    ap.add_argument("--paradigm", action="append",
                    help="paradigm name(s); repeatable. default: all")
    ap.add_argument("--n", type=int, default=4,
                    help="how many activities to use (default 4)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for --random sampling")
    ap.add_argument("--mock", action="store_true",
                    help="skip model load; emit canned paradigm text")
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--show-text", action="store_true",
                    help="print full generated text per item")
    ap.add_argument("--json", type=Path,
                    help="optional: write full results to JSON")
    args = ap.parse_args()

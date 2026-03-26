# Mixed Relay Selection Draft Report

## Headline
Most adult mixed relay teams are built from strong overall triathletes, using a combined individual-race lens that first checks the same event and then falls back to a related same-weekend race when needed.

## How To Read The Main Fields
| Field | Plain-English Meaning |
| --- | --- |
| individual_* | Combined individual-race comparison. It uses the same event first and falls back to a related same-weekend elite race if no exact individual result exists. |
| prior_365_* | Previous-365-day form ranking within country/gender, based on average normalized finish score where 1.0 is best in field and 0.0 is last in field. |
| individual_top2_slots | How many of the 4 relay athletes were top-2 within their country/gender under the combined individual-race comparison. |
| prior_365_top2_slots | How many of the 4 relay athletes were top-2 within their country/gender on prior-365-day form, even if they were not in that event's individual field. |

## Event-Class Summary
| event_class | teams | medal_teams | fully_individual_matched_teams | individual_slot_top2_rate | all_individual_top2_rate | prior_365_slot_top2_rate | avg_individual_country_rank | avg_prior_365_country_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| World Championship Level | 146 | 29 | 72 | 69.9% | 30.1% | 27.7% | 1.63 | 5.88 |
| WTCS | 79 | 18 | 37 | 65.2% | 24.1% | 19.6% | 1.65 | 6.67 |
| Continental Championship | 77 | 36 | 31 | 65.9% | 24.7% | 32.1% | 1.55 | 4.12 |
| Major Games | 57 | 13 | 28 | 77.2% | 38.6% | 40.8% | 1.44 | 4.90 |
| Other | 19 | 3 | 0 | 0.0% | 0.0% | 26.3% |  | 5.62 |
| Olympic Qualifier | 15 | 6 | 1 | 56.7% | 0.0% | 48.3% | 1.35 | 4.04 |
| Cup | 3 | 2 | 1 | 50.0% | 33.3% | 16.7% | 4.67 | 10.92 |
| Club Championship | 2 | 0 | 0 | 0.0% | 0.0% | 0.0% |  | 14.50 |

## Country Scorecard
| country | teams | medals | avg_finish | avg_individual_country_rank | avg_prior_365_country_rank | all_individual_top2_rate | all_prior_365_top2_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Great Britain | 17 | 12 | 3.41 | 1.74 | 9.52 | 23.5% | 0.0% |
| Germany | 19 | 12 | 3.63 | 1.90 | 9.90 | 47.4% | 0.0% |
| United States | 17 | 8 | 4.88 | 1.70 | 8.03 | 35.3% | 0.0% |
| France | 13 | 7 | 3.15 | 1.91 | 6.69 | 30.8% | 0.0% |
| Italy | 19 | 6 | 6.68 | 1.45 | 6.67 | 26.3% | 0.0% |
| Switzerland | 16 | 5 | 5.69 | 1.73 | 4.64 | 37.5% | 0.0% |
| New Zealand | 14 | 5 | 6.21 | 1.80 | 3.36 | 28.6% | 14.3% |
| China | 4 | 4 | 2.25 | 1.46 | 10.00 | 50.0% | 0.0% |
| Australia | 12 | 4 | 6.50 | 1.98 | 11.12 | 33.3% | 0.0% |
| Brazil | 13 | 4 | 7.54 | 1.48 | 3.87 | 46.2% | 7.7% |
| South Africa | 7 | 4 | 7.86 | 1.69 | 3.36 | 42.9% | 0.0% |
| Morocco | 3 | 3 | 1.33 | 1.56 | 2.33 | 0.0% | 0.0% |

## Medal Teams That Weren't Full Top-2 Rosters
| event_date | event_class | event_name | country | team_finish | individual_top2_slots | prior_365_top2_slots | avg_individual_country_rank | avg_prior_365_country_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-05-21 | Olympic Qualifier | 2021 World Triathlon Mixed Relay Olympic Qualification Event Lisbon | Belgium | 1 | 1 | 2 | 1.00 | 2.50 |
| 2025-04-05 | Continental Championship | 2025 Americas Triathlon Cup and South Americas Triathlon Championships Santiago | Ecuador | 2 | 2 | 2 | 1.00 | 3.00 |
| 2024-04-19 | Continental Championship | 2024 Africa Triathlon Championships Hurghada | Bulgaria | 2 | 1 | 1 | 1.00 | 1.00 |
| 2024-04-19 | Continental Championship | 2024 Africa Triathlon Championships Hurghada | Algeria | 3 | 1 | 2 | 1.00 | 2.33 |
| 2025-06-07 | Continental Championship | 2025 Americas Triathlon Championships Calima | United States | 1 | 3 | 0 | 1.33 | 8.50 |
| 2023-09-29 | Major Games | 2023 Hangzhou Asian Games | Japan | 1 | 3 | 0 | 1.33 | 9.25 |
| 2023-08-04 | World Championship Level | 2023 Europe Triathlon Sprint and Relay Championships Balikesir | Great Britain | 1 | 3 | 0 | 1.33 | 10.00 |
| 2023-06-27 | Major Games | 2023 Krakow-Malopolska European Games | Norway | 1 | 3 | 1 | 1.33 | 3.00 |
| 2022-10-29 | Continental Championship | 2022 Americas Triathlon Championships Montevideo | Brazil | 1 | 3 | 3 | 1.33 | 2.25 |
| 2023-09-29 | Major Games | 2023 Hangzhou Asian Games | China | 2 | 3 | 1 | 1.33 | 3.75 |
| 2022-09-17 | Continental Championship | 2022 Asia Triathlon Championships Aktau | Hong Kong, China | 2 | 3 | 1 | 1.33 | 3.50 |
| 2022-07-09 | WTCS | 2022 World Triathlon Championship Series Hamburg | Australia | 2 | 3 | 0 | 1.33 | 8.75 |

## Recurring Relay-Only Or Schedule-Managed Athletes
| relay_athlete | country | gender | relay_slots | events | last_event |
| --- | --- | --- | --- | --- | --- |
| Zsanett Kuttor-Bragmayer | Hungary | female | 6 | 6 | 2024-08-09 |
| Lotte Miller | Norway | female | 5 | 5 | 2024-07-30 |
| Oscar Gladney Rundqvist | Denmark | male | 5 | 5 | 2024-07-14 |
| Emil Holm | Denmark | male | 5 | 5 | 2023-08-20 |
| Mitch Kolkman | Netherlands | male | 5 | 5 | 2023-08-20 |
| Márta Kropkó | Hungary | female | 4 | 4 | 2025-07-12 |
| Miguel Hidalgo | Brazil | male | 4 | 4 | 2023-08-20 |
| Nicolò Strada | Italy | male | 4 | 4 | 2023-08-20 |
| Barbara De Koning | Netherlands | female | 3 | 3 | 2025-07-12 |
| Carlotta Missaglia | Italy | female | 3 | 3 | 2025-07-12 |
| Nicola Azzano | Italy | male | 3 | 3 | 2025-02-15 |
| Samuel Dickinson | Great Britain | male | 3 | 3 | 2024-07-30 |

## Draft Slide Structure
- Slide 1: Headline and method. Combined individual-race check first, prior-365 country form second.
- Slide 2: Event-class summary using the combined individual-race columns as the primary measure.
- Slide 3: Country scorecard using average individual-race rank and all-individual-top2 rate.
- Slide 4: Exceptions and roster strategy, including relay-only or schedule-managed athletes.
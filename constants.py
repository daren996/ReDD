SPIDER_DN_LIST = ["store_1", "wine_1", "college_2", "flight_4"]  # , "soccer_1"

SPIDER_DN2FN = {
  "store_1": ["albums", "employees", "customers", "invoices", "tracks"],
  "wine_1": ["grapes", "appellations", "wine"],
  "soccer_1": ["player_attributes", "team_attributes", "player", "league", "team"],
  "bike_1": [],
  "apartment_rentals": [],
  "college_2": ["classroom", "section", "takes", "department", "course", "teaches", "student", "instructor",
                "advisor", "prereq"],
  "flight_4": ["routes", "airports", "airlines"]
}

EXP_DN2FN = {
  ## spider dataset
  "store_1": ["albums", "customers", "invoices", "tracks",
              "customers-invoices", "customers-employees", "albums-tracks",
              "employees-customers-invoices"],
  "wine_1": ["wine-grapes", "wine-appellations"],  
  # "grapes", "appellations", "wine"
  "soccer_1": ["player_attributes", "team_attributes", "player", "league", "team", "all"],
  "bike_1": ["all"],
  "apartment_rentals": ["all"],
  "college_2": ["course-section", "instructor-teaches", "course-teaches", "instructor-department",
                "department-student-instructor", "course-teaches-instructor", "course-section-classroom"],
  # "classroom", "section", "takes", "department", "course", "teaches", "student", "instructor", "advisor", "prereq"
  "flight_4": ["routes-airports", "routes-airlines","routes-airports-airlines"],
  # "routes", "airports", "airlines"
  ## bird dataset
  "california_schools": ["all"],
  "debit_card_specializing": ["all"],
  "student_club": ["all"],
  ## galois dataset
  "fortune": ["all"],
  "premierleague": ["all"]
}

ASSIGN_THRESHOLD = 5

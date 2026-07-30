export const AGE_GROUP_FILTERS = [
  { value: "", label: "Any age group", minAge: "", maxAge: "" },
  { value: "child", label: "Child", minAge: "0", maxAge: "12" },
  { value: "teenager", label: "Teenager", minAge: "13", maxAge: "17" },
  { value: "adult", label: "Adult", minAge: "18", maxAge: "59" },
  { value: "older_adult", label: "Senior", minAge: "60", maxAge: "120" },
];

export function formatAgeGroup(value) {
  const age = Number(value);
  if (!Number.isFinite(age) || age <= 0) return "Unknown";
  if (age <= 12) return "Child";
  if (age <= 17) return "Teenager";
  if (age <= 59) return "Adult";
  return "Senior";
}

export function selectedAgeGroup(filters) {
  return (
    AGE_GROUP_FILTERS.find(
      (group) => group.minAge === filters.min_age && group.maxAge === filters.max_age,
    )?.value || ""
  );
}

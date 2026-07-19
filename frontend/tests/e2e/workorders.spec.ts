import { test, expect, type APIRequestContext } from "@playwright/test";

const apiBase = process.env.CI ? "http://127.0.0.1:8000" : "http://127.0.0.1:8010";

async function routeApiToIsolatedBackend(page: import("@playwright/test").Page) {
  await page.route("http://localhost:8000/**", async (route) => {
    const source = new URL(route.request().url());
    await route.continue({ url: `${apiBase}${source.pathname}${source.search}` });
  });
}

async function createDraft(request: APIRequestContext, asset: string) {
  const response = await request.post(`${apiBase}/api/v1/workorders`, {
    data: {
      asset_id: asset,
      failure_mode: "QA isolated bearing regression",
      recommended_action: `Inspect ${asset}; remain pending until explicit human decision.`,
      rul_estimate: 12.5,
      anomaly_score: 0.88,
      manual_refs: ["SOP-PM-006"],
      priority: "critical",
      estimated_duration_hrs: 4,
    },
  });
  expect(response.status()).toBe(201);
  return response.json();
}

test.describe("isolated human approval workflow", () => {
  test("approves and rejects separate pending drafts", async ({ page, request }) => {
    await routeApiToIsolatedBackend(page);
    const approvedDraft = await createDraft(request, "ISM-CMR-004");
    await page.goto(`/workorders/${approvedDraft.id}`);
    await expect(page.getByText("PENDING APPROVAL")).toBeVisible();
    await page.getByLabel(/Assign Technician/i).fill("TECH-E2E-01");
    await page.getByLabel(/Proposed Start/i).fill("2026-07-22T09:30");
    await page.getByRole("button", { name: "Submit Decision" }).click();
    await expect(page.getByText("Decision Logged")).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText(/approved and moved/i)).toBeVisible();

    const rejectedDraft = await createDraft(request, "ISM-GBX-003");
    await page.goto(`/workorders/${rejectedDraft.id}`);
    await page.getByRole("button", { name: "Reject Order" }).click();
    await page.getByLabel(/Reason for Rejection/i).fill("Isolated QA rejection");
    await page.getByRole("button", { name: "Submit Decision" }).click();
    await expect(page.getByText(/was rejected/i)).toBeVisible({ timeout: 10_000 });

    await page.goto("/workorders");
    await page.getByRole("combobox").first().selectOption("PENDING_APPROVAL");
    await expect(page.getByRole("row").filter({ hasText: `#${approvedDraft.id}` })).toHaveCount(0);
    await expect(page.getByRole("row").filter({ hasText: `#${rejectedDraft.id}` })).toHaveCount(0);
  });
});

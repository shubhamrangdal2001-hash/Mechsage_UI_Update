// frontend/tests/e2e/fleet.spec.ts
import { test, expect } from "@playwright/test";

test.describe("MechSage Fleet Overview E2E Test", () => {
  test("should load the fleet dashboard and display key telemetry", async ({ page }) => {
    // 1. Visit the root page (should redirect to /fleet)
    await page.goto("/");
    await expect(page).toHaveURL(/.*\/fleet/);

    // 2. Verify Page Header is visible
    await expect(
      page.getByRole("heading", { name: "Fleet command center" })
    ).toBeVisible();

    // 3. Verify KPI cards are rendered
    await expect(page.getByText("Assets online")).toBeVisible();
    await expect(page.getByText("Current cycle")).toBeVisible();

    // 4. Verify telemetry logs section
    await expect(page.getByRole("heading", { name: "Incident log" })).toBeVisible();
  });

  test("should allow navigating to engine details", async ({ page }) => {
    await page.goto("/fleet");

    // Public UI must expose the Ironside unit, never its internal FD mapping.
    const engineCard = page.getByRole("link", { name: /ISM-CNC-001/ });
    await expect(engineCard).toBeVisible();
    await engineCard.click();

    // Verify redirect to detail page
    // Dataset routing stays internal while the page renders the public asset ID.
    await expect(page).toHaveURL(/.*\/engines\/FD001/);

    // Verify detail elements
    await expect(page.getByRole("heading", { name: /ISM-CNC-001/ })).toBeVisible();
    await expect(page.getByText(/RUL Projection/i)).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Anomaly Score Profile" })
    ).toBeVisible();
  });
});

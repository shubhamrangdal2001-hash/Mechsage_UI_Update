import { test, expect } from "@playwright/test";

test.describe("RAG guardrails", () => {
  test("returns Ironside gear-pitting evidence and abstains off-domain", async ({ page }) => {
    const apiBase = process.env.CI ? "http://127.0.0.1:8000" : "http://127.0.0.1:8010";
    await page.route("http://localhost:8000/**", async (route) => {
      const source = new URL(route.request().url());
      await route.continue({ url: `${apiBase}${source.pathname}${source.search}` });
    });
    await page.goto("/rag");
    const query = page.getByRole("textbox");
    await query.fill("What evidence and procedure applies to ISM-GBX-003 gear pitting?");
    await page.getByRole("button", { name: /query|search|retrieve/i }).click();
    await expect(page.getByText("Engineered Response")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/ISM-GBX-003/).first()).toBeVisible();
    await expect(page.getByText(/Ironside Manufacturing knowledge base/).first()).toBeVisible();
    await expect(page.getByText(/turbofan unit/i)).toHaveCount(0);

    await query.fill("renaissance sonnet meter");
    await page.getByRole("button", { name: /query|search|retrieve/i }).click();
    await expect(page.getByText("Guardrail Triggered")).toBeVisible({ timeout: 15_000 });
  });
});

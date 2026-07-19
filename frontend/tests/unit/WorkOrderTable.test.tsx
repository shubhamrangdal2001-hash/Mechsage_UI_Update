import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { WorkOrderTable } from "@/components/organisms/WorkOrderTable";
import { pendingWorkOrder } from "./fixtures";

describe("WorkOrderTable", () => {
  it("filters pending, approved and priority rows", () => {
    const approved = { ...pendingWorkOrder, id: 2, asset_id: "ISM-HYD-002", status: "approved" as const, priority: "low" as const };
    render(<WorkOrderTable workOrders={[pendingWorkOrder, approved]} />);
    const selects = screen.getAllByRole("combobox");
    fireEvent.change(selects[0], { target: { value: "PENDING_APPROVAL" } });
    expect(screen.getByText("ISM-CNC-001")).toBeInTheDocument();
    expect(screen.queryByText("ISM-HYD-002")).not.toBeInTheDocument();
    fireEvent.change(selects[1], { target: { value: "LOW" } });
    expect(screen.getByText(/No work orders matching/)).toBeInTheDocument();
  });

  it("never shows approved work under Pending Approval", () => {
    const approved = { ...pendingWorkOrder, id: 2, status: "approved" as const };
    render(<WorkOrderTable workOrders={[approved]} />);
    fireEvent.change(screen.getAllByRole("combobox")[0], { target: { value: "PENDING_APPROVAL" } });
    expect(within(screen.getByRole("table")).queryByText("APPROVED")).not.toBeInTheDocument();
  });
});

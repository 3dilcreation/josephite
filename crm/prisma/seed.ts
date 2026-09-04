import { randomBytes } from "node:crypto";
import { PrismaClient } from "@prisma/client";
import bcrypt from "bcryptjs";

const db = new PrismaClient();

const daysFromNow = (days: number) => {
  const date = new Date();
  date.setDate(date.getDate() + days);
  return date;
};

async function main() {
  const adminEmail = process.env.SEED_ADMIN_EMAIL ?? "admin@3dilcreation.com";
  const adminPassword = process.env.SEED_ADMIN_PASSWORD ?? "ChangeMe123!";

  const org = await db.organization.upsert({
    where: { slug: "3dil-creation" },
    update: {},
    create: { name: "3DIL CREATION", slug: "3dil-creation", currency: "INR" },
  });

  const hq = await db.branch.upsert({
    where: { orgId_code: { orgId: org.id, code: "HQ" } },
    update: {},
    create: { orgId: org.id, name: "3DIL CREATION — Head Office", code: "HQ", type: "HQ" },
  });

  const departmentSeeds = [
    { name: "Sales & Marketing", slug: "sales-marketing", colour: "#ef6820", description: "Leads, quotes, follow-ups" },
    { name: "Design", slug: "design", colour: "#7c3aed", description: "CAD, file repair, print preparation" },
    { name: "Production", slug: "production", colour: "#0284c7", description: "Printing, post-processing, QC" },
    { name: "Dispatch & Support", slug: "dispatch-support", colour: "#059669", description: "Packing, shipping, after-sales" },
    { name: "Accounts", slug: "accounts", colour: "#64748b", description: "Invoicing and collections" },
  ];

  const departments = await Promise.all(
    departmentSeeds.map((department) =>
      db.department.upsert({
        where: { orgId_slug: { orgId: org.id, slug: department.slug } },
        update: {},
        create: { orgId: org.id, ...department },
      }),
    ),
  );

  const byslug = (slug: string) => departments.find((d) => d.slug === slug)!;

  const admin = await db.user.upsert({
    where: { email: adminEmail },
    update: {},
    create: {
      orgId: org.id,
      branchId: hq.id,
      email: adminEmail,
      name: "3DIL Admin",
      role: "SUPER_ADMIN",
      passwordHash: await bcrypt.hash(adminPassword, 10),
    },
  });

  const demoPassword = await bcrypt.hash("Demo1234!", 10);
  const staffSeeds = [
    { email: "sales@3dilcreation.com", name: "Sales Executive", role: "STAFF" as const, dept: "sales-marketing" },
    { email: "design@3dilcreation.com", name: "Design Lead", role: "MANAGER" as const, dept: "design" },
    { email: "production@3dilcreation.com", name: "Production Manager", role: "MANAGER" as const, dept: "production" },
    { email: "accounts@3dilcreation.com", name: "Accounts", role: "VIEWER" as const, dept: "accounts" },
  ];

  const staff = await Promise.all(
    staffSeeds.map(async (person) =>
      db.user.upsert({
        where: { email: person.email },
        update: {},
        create: {
          orgId: org.id,
          branchId: hq.id,
          email: person.email,
          name: person.name,
          role: person.role,
          passwordHash: demoPassword,
          departments: { create: [{ departmentId: byslug(person.dept).id }] },
        },
      }),
    ),
  );

  const [salesUser, designUser, productionUser] = staff;

  const websiteSource = await db.integrationSource.upsert({
    where: { orgId_key: { orgId: org.id, key: "website-quote-form" } },
    update: {},
    create: {
      orgId: org.id,
      key: "website-quote-form",
      name: "3dilcreation.com quote form",
      kind: "WEBSITE",
      secret: randomBytes(24).toString("hex"),
      defaultDepartmentId: byslug("sales-marketing").id,
      defaultBranchId: hq.id,
    },
  });

  await db.integrationSource.upsert({
    where: { orgId_key: { orgId: org.id, key: "indiamart-push" } },
    update: {},
    create: {
      orgId: org.id,
      key: "indiamart-push",
      name: "IndiaMART enquiries",
      kind: "INDIAMART",
      secret: randomBytes(24).toString("hex"),
      defaultDepartmentId: byslug("sales-marketing").id,
      defaultBranchId: hq.id,
    },
  });

  await db.integrationSource.upsert({
    where: { orgId_key: { orgId: org.id, key: "whatsapp-business" } },
    update: {},
    create: {
      orgId: org.id,
      key: "whatsapp-business",
      name: "WhatsApp Business",
      kind: "WHATSAPP",
      secret: randomBytes(24).toString("hex"),
      defaultDepartmentId: byslug("sales-marketing").id,
      defaultBranchId: hq.id,
    },
  });

  if ((await db.customer.count({ where: { orgId: org.id } })) === 0) {
    const acme = await db.customer.create({
      data: {
        orgId: org.id,
        branchId: hq.id,
        name: "Ravi Kumar",
        company: "Acme Automation Pvt Ltd",
        email: "ravi@acmeautomation.in",
        phone: "9876543210",
        gstin: "33AABCA1234F1Z5",
        city: "Chennai",
        state: "Tamil Nadu",
        ownerId: salesUser.id,
      },
    });

    const studio = await db.customer.create({
      data: {
        orgId: org.id,
        branchId: hq.id,
        name: "Priya Menon",
        company: "Studio Nine Architects",
        email: "priya@studionine.in",
        phone: "9845012345",
        city: "Bengaluru",
        state: "Karnataka",
        ownerId: salesUser.id,
      },
    });

    await db.lead.createMany({
      data: [
        {
          orgId: org.id,
          branchId: hq.id,
          departmentId: byslug("sales-marketing").id,
          assignedToId: salesUser.id,
          title: "200 × PA12 cable clips, MJF",
          contactName: "Anand S",
          company: "Vertex Motors",
          phone: "9900112233",
          requirement: "Repeat monthly batch. Needs a fixed per-unit rate and a 5-day turnaround.",
          status: "QUOTED",
          priority: "HIGH",
          channel: "ONLINE",
          sourceKind: "INDIAMART",
          estimatedValue: 48000,
          expectedCloseDate: daysFromNow(6),
        },
        {
          orgId: org.id,
          branchId: hq.id,
          departmentId: byslug("sales-marketing").id,
          title: "Architectural model, 1:200 site",
          contactName: "Priya Menon",
          company: "Studio Nine Architects",
          phone: "9845012345",
          customerId: studio.id,
          requirement: "White resin, presentation finish, delivery before the client review.",
          status: "NEGOTIATION",
          priority: "URGENT",
          channel: "ONLINE",
          sourceKind: "WEBSITE",
          sourceId: websiteSource.id,
          estimatedValue: 95000,
          expectedCloseDate: daysFromNow(3),
        },
        {
          orgId: org.id,
          branchId: hq.id,
          title: "Prosthetic socket trial print",
          contactName: "Dr. Nithya",
          phone: "9812345678",
          requirement: "Single trial piece, biocompatible resin if possible.",
          status: "NEW",
          priority: "MEDIUM",
          channel: "OFFLINE",
          sourceKind: "REFERRAL",
          estimatedValue: 12000,
        },
      ],
    });

    const order1 = await db.order.create({
      data: {
        orgId: org.id,
        branchId: hq.id,
        orderNo: `3DIL-${new Date().getFullYear()}-0001`,
        customerId: acme.id,
        departmentId: byslug("production").id,
        assignedToId: productionUser.id,
        status: "PRINTING",
        priority: "HIGH",
        channel: "OFFLINE",
        sourceKind: "WALK_IN",
        dueDate: daysFromNow(2),
        paymentDueDate: daysFromNow(9),
        taxAmount: 6480,
        shipping: 500,
        items: {
          create: [
            {
              name: "Assembly jig v4",
              technology: "FDM",
              material: "PETG-CF",
              colour: "Black",
              quantity: 12,
              unitPrice: 2400,
              weightGrams: 180,
              printHours: 6.5,
              fileName: "jig_v4.stl",
            },
            {
              name: "Locating pin",
              technology: "SLA",
              material: "Tough resin",
              quantity: 24,
              unitPrice: 300,
              printHours: 3,
            },
          ],
        },
      },
    });

    await db.payment.create({
      data: {
        orderId: order1.id,
        amount: 20000,
        method: "UPI",
        reference: "UPI/4411290034",
        recordedById: admin.id,
      },
    });

    const order2 = await db.order.create({
      data: {
        orgId: org.id,
        branchId: hq.id,
        orderNo: `3DIL-${new Date().getFullYear()}-0002`,
        customerId: studio.id,
        departmentId: byslug("design").id,
        assignedToId: designUser.id,
        status: "DESIGN",
        priority: "URGENT",
        channel: "ONLINE",
        sourceKind: "WEBSITE",
        sourceId: websiteSource.id,
        dueDate: daysFromNow(-1),
        paymentDueDate: daysFromNow(-3),
        taxAmount: 8100,
        items: {
          create: [
            {
              name: "Site model, 1:200",
              technology: "SLA",
              material: "Standard resin",
              colour: "White",
              quantity: 1,
              unitPrice: 45000,
              printHours: 26,
              fileName: "site_1_200.stl",
            },
          ],
        },
      },
    });

    // Totals are always derived, never typed in — recompute both seeded orders.
    for (const orderId of [order1.id, order2.id]) {
      const order = await db.order.findUnique({
        where: { id: orderId },
        include: { items: true, payments: true },
      });
      if (!order) continue;

      const subtotal = order.items.reduce(
        (sum, item) => sum + Number(item.unitPrice) * item.quantity,
        0,
      );
      const total =
        subtotal + Number(order.taxAmount) + Number(order.shipping) - Number(order.discount);
      const paid = order.payments.reduce((sum, payment) => sum + Number(payment.amount), 0);
      const overdue = order.paymentDueDate ? order.paymentDueDate < new Date() : false;

      await db.order.update({
        where: { id: orderId },
        data: {
          subtotal,
          total,
          amountPaid: paid,
          paymentStatus:
            paid <= 0 ? (overdue ? "OVERDUE" : "UNPAID") : paid >= total ? "PAID" : overdue ? "OVERDUE" : "PARTIAL",
        },
      });
    }

    await db.notification.create({
      data: {
        orgId: org.id,
        userId: admin.id,
        type: "ORDER_OVERDUE",
        title: `Order 3DIL-${new Date().getFullYear()}-0002 is past its delivery date`,
        body: "Still in Design. Studio Nine Architects.",
        link: `/orders/${order2.id}`,
      },
    });
  }

  console.log(`Seeded. Sign in as ${adminEmail} / ${adminPassword}`);
  console.log("Demo staff logins use the password Demo1234!");
}

main()
  .catch((error) => {
    console.error(error);
    process.exit(1);
  })
  .finally(() => db.$disconnect());

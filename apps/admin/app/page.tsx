import { redirect } from "next/navigation";

/** The staff product always starts at its dedicated sign-in screen. */
export default function Home() {
  redirect("/login");
}

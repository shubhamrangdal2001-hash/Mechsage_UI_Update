import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: false, // prevent double useEffect firing in dev (synthetic engine)
};

export default nextConfig;

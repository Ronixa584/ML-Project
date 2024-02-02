import React from "react";
import { Header } from "./Header";
import Footer from "./Footer";
import Banner from "./Banner";
import About from "./About";
import AboutIII from "./AboutIII";
import Location from "./Location";
import AboutDeveloper from "./AboutDevloper";
import Overview from "./Overview";
import ImageSlider from "./ImageSlider";
import FloorPlan from "./FloorPlan";

export const Home = () => {
  return (
    <div>
      <Header />
      <Banner />
      <About />
      <Overview />
      <AboutIII />
      <ImageSlider />
      <FloorPlan />
      <Location />
      <AboutDeveloper />
      <Footer />
    </div>
  );
};

export default Home;

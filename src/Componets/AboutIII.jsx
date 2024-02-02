import React from "react";

const AboutIII = () => {
  return (
    <section className="pt-[40px] pb-[40px] flex flex-col px-6 lg:flex-row lg:justify-center ">
      <div className="lg:w-[1100px] lg:h-[428px] lg:mr-5 lg:ml-[100px] xl:mr-10 xl:ml-[181px]">
        <img
          src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/capsule-lift.jpg"
          width="100%"
          style={{ borderRadius: "17px" }}
          alt="Overview"
          className="h-full"
        />
      </div>
      <div className="lg:mr-[60px] xl:mr-[170px] ">
        <h2
          className=""
          style={{
            textAlign: "left",
            color: "#242424",
            marginBottom: "0px",
            fontFamily: "dayana Italic",
            fontSize: "50px",
          }}
        >
          High Living
        </h2>
        <h4>
          {" "}
          <b>A PREVIEW OF YOUR LIFESTYLE</b>
        </h4>
        <h5>
          <b>FEATURES</b>
        </h5>
        <ul>
          <li>Two Fully Automatic High Speed Lifts</li>
          <li>Boom Barrier Systems at Entry / Exit Points</li>
          <li>An Integrated Fire Fighting System</li>
          <li>Vertical Garbage Shafts</li>
          <li>Sewerage Treatment Plant</li>
          <li>Generator Backup for Essential Services</li>
          <li>Automated Parking Tower</li>
        </ul>
        <br />
        <p>
          Enhancing the breath-taking architectural aesthetics of Woods Platina
          is a Capsule Lift in Walnut Tower with striking interiors and large
          glass viewing panels overlooking the beautiful recreational spaces
          bound by the scenic skyline
        </p>
        <br />
        <a className="bg-[#6e5e31] text-white p-[10px]">Know More</a>
      </div>
    </section>
  );
};

export default AboutIII;

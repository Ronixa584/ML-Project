import React from "react";

const About = () => {
  return (
    <section
      id="AboutID"
      className="pt-[40px] pb-[40px] flex flex-col px-6 lg:flex-row lg:justify-center "
    >
      <div className="lg:w-[720px] lg:h-[428px] lg:mr-5 lg:ml-[100px]  xl:mr-10  xl:ml-[181px]">
        <img
          src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/about1.jpg"
          width="100%"
          style={{ borderRadius: "17px" }}
          alt="Overview"
          className="h-full"
        />
      </div>
      <div className="lg:mr-[60px] xl:mr-[170px] ">
        <h4>
          Introducing
          <br />2 &amp; 3 BHK
        </h4>
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
          Ultra Luxury Lifespaces
        </h2>
        <h4 style={{ marginBottom: "30px" }}>
          IN WALNUT &amp; OAK @ WOODS PLATINA
        </h4>
        <p style={{ color: "#000" }}>
          After the grand success of Maple &amp; Ebony Towers, SMGK Group has to
          offer you Ultra Luxury lifespaces in Walnut &amp; Oak Towers @ Woods
          Platina
          <br />
          <br />
          Curated with smart technology features and modern amenities, these
          thoughtfully designed residences make your life Simpler, Smarter and
          Efficient
          <br />
          <br />
          In a congested city like Mumbai only a privileged few get the
          opportunity to enjoy Nature and Luxury in one place
          <br />
          <br />
          So make the most out of it and get the best of both the worlds only at
          - Woods Platina
        </p>
        <br />
        <a className="bg-[#6e5e31] text-white p-[10px]">Know More</a>
      </div>
    </section>
  );
};

export default About;

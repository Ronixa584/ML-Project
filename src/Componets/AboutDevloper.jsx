import React from "react";
import { useState } from "react";
import RF1 from "./RF1";

const AboutDeveloper = () => {

    const [ModalOpen, setModalOpen] = useState(false);

    const openModal = () => {
      setModalOpen(true);
    };

    const closeModal = () => {
      setModalOpen(false);
  };
  
  return (
    <div
      id="AboutDeveloperID"
      className="mb-[60px] md:mb-[105px] lg:mb-[75px] xl:mb-[50px]"
    >
      <section
        className=""
        id="sec-about"
        style={{
          padding: "40px 0 10px 0",
          textAlign: "center",
          background: "#133a1f",
        }}
      >
        <div className="flex flex-col justify-center items-center">
          <div
            className="text-[15px] m-[16px] "
          >
            <h2 className="wow slideInDown headertext text-white">
              About Developer
            </h2>
            <div className="flex justify-center items-center">
              <img
                className="hidden-lg hidden-md"
                src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/developer-logo.png"
                style={{ width: "13%" }}
                alt="Developer Logo"
              />
            </div>
            <br />
            <br />
            <p style={{ textAlign: "center", color: "#fff" }}>
              Our mission towards contributing to a better world began in the
              year 2000, when SMGK Group was conceived by our Founder Chairman
              and Group President, Mr. Mohammad Ali Momin
            </p>
            <p style={{ textAlign: "center", color: "#fff" }}>
              With over four decades of industry experience and above all, a
              strong Conscience, Mr.Momin strongly defined the Ethos of the
              Group - to be agents of Social change
            </p>
            <p style={{ textAlign: "center", color: "#fff" }}>
              What's more, we have set benchmarks in the Indian Real Estate
              Industry by employing the best Environmental practices and
              ensuring use of Smart
            </p>
            <br />
            <p
              style={{
                textAlign: "center",
                color: "#fff",
                fontSize: "20px",
                fontWeight: "600",
              }}
            >
              Above all, we have Built with Conscience - a Legacy of Trust,
              Quality and Social Responsibility
            </p>
            <br />
          </div>

          <div className="lg:flex text-white">
            <div className="col-md-4">
              <div style={{ marginBottom: "25px" }}>
                <span className="font-extrabold">1+ MILLION</span>
                <br />
                <span className="deve-text1">Sq Ft Of Development</span>
              </div>
            </div>
            <div className="col-md-4">
              <div style={{ marginBottom: "25px" }}>
                <span className="font-extrabold">4+ DECADES</span>
                <br />
                <span className="deve-text1"> Of Legacy</span>
              </div>
            </div>
            <div className="col-md-4">
              <div style={{ marginBottom: "25px" }}>
                <span className="font-extrabold">1000+</span>
                <br />
                <span className="deve-text1">Happy Customers</span>
              </div>
            </div>
          </div>
          <br />

          <p style={{ fontSize: "13px", margin: "16px", color: "#fff" }}>
            MahaRERA Registration Number WALNUT : P51800048890 | OAK :
            P51800049102 | Ebony : P51800009546 and is available on the website
            https://maharera.mahaonline.gov.in.
          </p>

          <div className="row">
            <div className="lg:flex">
              <div className="col-md-3" align="center">
                <img
                  src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/walnut.jpg"
                  style={{ borderRadius: 0 }}
                  alt="Walnut Property"
                />
                <p style={{ color: "#fff", fontSize: "14px" }}>
                  WALNUT : P51800048890
                </p>
              </div>
              <div className="col-md-3" align="center">
                <img
                  src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/oak.jpg"
                  style={{ borderRadius: 0 }}
                  alt="Oak Property"
                />
                <p style={{ color: "#fff", fontSize: "14px" }}>
                  OAK : P51800049102
                </p>
              </div>
            </div>
          </div>
        </div>
        <br />
      </section>
      <div className="bg-white pt-4 pb-4 flex flex-col justify-center items-center">
        <img
          src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/Anarock-logoB1.png"
          className="w-[120px]"
        />

        <br />

        <span className="text-black">
          {" "}
          Anarock Property Consultants Private Limited, MAHARERA registration
          no. A51900000108.
        </span>
      </div>
      <div className="fixed top-1/2 right-[-73px]">
        <button
          onClick={openModal}
          className="bg-[#6e5e31] text-white px-4 py-2 -rotate-90 p-2"
        >
          Download Brochure
        </button>
      </div>

      {/* Modal */}
      <>
        {ModalOpen ? (
          <>
            <RF1 isOpen={ModalOpen} onClose={closeModal} />
          </>
        ) : (
          <></>
        )}
      </>
    </div>
  );
};

export default AboutDeveloper;

import React from "react";

const Location = () => {
  return (
    <section id="LocationID" className="pt-[40px]">
      {/* Title Location */}
      <div className="px-6 ">
        <div className="flex justify-center mt-[20px] mb-[40px] text-[27px] font-bold">
          LOCATION
        </div>
        <div className="flex justify-center mb-[16px] text-[16px]">
          The strategic location of Woods Platina is one of the main reasons why
          the project has immense scope for Value appreciation over time.
          Considering the fact that your home will be worth substantially more a
          few years down the line, investing in Woods Platina is a great idea...
        </div>
      </div>

      {/* map */}
      <div className="flex items-center justify-center px-6 ">
        <div className="row">
          <div className="mb-[15px]">
            <img
              src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/map.jpg"
              width="100%"
              className="lg:w-[540px] lg:h-[500px]"
            />
          </div>
        </div>
      </div>

      {/* Routes */}
      <div className="flex flex-col">
        <div className="" style={{ background: "#e7e8ea", marginTop: "27px" }}>
          <div className="lg:flex lg:flex-row">
            <LocationInfo
              imageSrc="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/l1.jpg"
              locText="2 MINUTE"
              locPara="Walk From The Nearest Metro Station"
            />
            <LocationInfo
              imageSrc="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/l2.jpg"
              locText="8 MINUTES"
              locPara="Away From Lokhandwala Complex"
            />
            <LocationInfo
              imageSrc="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/l3.jpg"
              locText="10 MINUTES"
              locPara="Walk From The Nearest Railway Station"
            />
            <LocationInfo
              imageSrc="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/l4.jpg"
              locText="4 MINUTES"
              locPara="Drive To Link Road"
            />
          </div>
        </div>

        <div
          className=""
          style={{ background: "#e7e8ea", paddingBottom: "40px" }}
        >
          <div className="lg:flex lg:flex-row lg:justify-center lg:items-center">
            <LocationInfo
              imageSrc="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/l5.jpg"
              locText="5 MINUTES"
              locPara="Drive To Western Express Highway"
            />
            <LocationInfo
              imageSrc="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/l6.jpg"
              locText="2 MINUTES"
              locPara="Walk from 12.5 Acres Veera Desai Sports Complex"
            />
            <LocationInfo
              imageSrc="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/l7.jpg"
              locText="20 MINUTES"
              locPara="Drive To Domestic And International Airports"
            />
            <LocationInfo
              imageSrc="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/l8.jpg"
              locText="15 MINUTES"
              locPara="Drive to Kokila Ben Dhirubhai Ambani Hospital"
            />
          </div>
        </div>
      </div>

      <div
        className="row"
        style={{ background: "#e7e8ea", paddingBottom: "40px" }}
      >
        <div className="col-md-3"></div>
        <div className="col-md-6">
          <div style={{ textAlign: "center", fontWeight: 600 }}>
            Site Address : <br />
            Sales Experience Centre - Woods Platina, New JVLR Extension Road,
            Near Evershine Embassy, Oshivara, Mumbai - 400 102
          </div>
        </div>
        <div className="col-md-3"></div>
      </div>
    </section>
  );
};

export default Location;

const LocationInfo = ({ imageSrc, locText, locPara }) => {
  return (
    <div className="flex flex-row border w-full">
      <div className="">
        <img src={imageSrc} className="" alt="Location" />
      </div>
      <div className="">
        <span className="">{locText}</span>
        <p className="">{locPara}</p>
      </div>
    </div>
  );
};

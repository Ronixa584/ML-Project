import React from "react";

const Overview = () => {
  return (
    <section className="mt-[30px] flex flex-col px-6 lg:flex-row">
      <div className="container">
        <div
          className="lg:mr-5 lg:ml-[100px] xl:mr-10 xl:ml-[181px]"
          style={{}}
        >
          <div className="col-md-7" style={{ marginTop: "0px" }}>
            <h4 style={{}}>
              Luxury
              <br />A KING SIZE LIFE OF
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
              Urban Indulgences
            </h2>
            <p style={{ color: "#000", textTransform: "uppercase" }}>
              At Woods Platina, You have an exclusive rooftop garden with a host
              of amenities for you to savor the high life
            </p>
            <div className="grid grid-cols-2">
              <div
                className="col-md-6 col-xs-6 col-sm-6"
                style={{ padding: "0px" }}
              >
                <div className="ame">
                  <img src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/i1.jpg" />
                  <span>Exquisite Sky Garden</span>
                </div>
                <br />
                <div className="ame">
                  <img src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/i2.jpg" />
                  <span>Children's Play Area</span>
                </div>
                <br />
                <div className="ame">
                  <img src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/i3.jpg" />
                  <span>Sit-Out Areas</span>
                </div>
              </div>
              <div
                className="col-md-6 col-xs-6 col-sm-6"
                style={{ padding: "0px" }}
              >
                <div className="ame">
                  <img src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/i4.jpg" />
                  <span>Jogging Track</span>
                </div>
                <br />
                <div className="ame">
                  <img src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/i5.jpg" />
                  <span>Infinity Edge Pool</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="col-md-5">
        <div class="w-auto lg:w-[450px] lg:h-[428px] lg:ml-5 lg:mr-[100px] xl:ml-10 xl:mr-[181px] rounded-md">
          <img
            src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/ame.jpg"
            width="100%"
            style={{ borderRadius: "17px" }}
          />
        </div>
        <div class="text-center">Rooftop Amenities</div>
        <div class=" text-center">Rooftop Amenities</div>
      </div>
    </section>
  );
};

export default Overview;

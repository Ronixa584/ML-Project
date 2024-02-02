import React from "react";
import { Swiper, SwiperSlide } from "swiper/react";
import "swiper/css";
import "swiper/css/navigation";
import "swiper/css/pagination";
import "swiper/swiper-bundle.css";
import {
  Navigation,
  Pagination,
  Mousewheel,
  Keyboard,
  Autoplay,
} from "swiper/modules";
import "./Slider.css";

const ImageSlider = () => {
  const slides = [
    {
      title1: "Grand Entrance Foyer 25ft High",
      image1:
        "https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/g1.jpg",
      title2: "Double Highted Air-Con Lobby",
      image2:
        "https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/g2.jpg",
    },
    {
      title1: "Master Bedroom",
      image1:
        "https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/g3.jpg",
      title2: "Kitchen",
      image2:
        "https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/g4.jpg",
    },
    {
      title1: "Kid's Room",
      image1:
        "https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/g5.jpg",
      title2: "Bathroom",
      image2:
        "https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/g6.jpg",
    },
  ];
  return (
    <div id="ImageSliderID">
      <div className="Laptop bg-[#6e5e312b]">
        <div className="text-center mb-5 pt-10 font-extrabold text-xl">
          Project Gallery
        </div>
        <MySwiper slides={slides} />
      </div>
    </div>
  );
};

export default ImageSlider;

// Reusable Swiper Component
const MySwiper = ({ slides }) => {
  return (
    <div className="relative overflow-hidden p-12">
      <Swiper
        cssMode={true}
        navigation={true}
        mousewheel={true}
        keyboard={true}
        modules={[Navigation, Pagination, Mousewheel, Keyboard, Autoplay]}
        autoplay={{ delay: 5000 }}
        speed={2000}
        className=""
      >
        {slides.map((slide, index) => (
          <SwiperSlide
            key={index}
            className="flex flex-col lg:flex-row items-center justify-center gap-3 "
          >
            {/* <div className=""> */}
            <div
              className="w-[420px] lg:w-[560px] h-[330px] border border-black bg-white flex xl:gap-2 justify-center items-center"
              style={{
                boxShadow:
                  "0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22)",
              }}
            >
              <div>
                <div className="">{slide.title1}</div>
                <img
                  src={slide.image1}
                  alt={`Slide ${index + 1}`}
                  className="w-[400px] h-[225px] object-cover"
                />
              </div>
            </div>
            <div
              className="hidden w-[420px] lg:w-[560px] h-[330px] border border-black bg-white lg:flex xl:gap-2 justify-center items-center"
              style={{
                boxShadow:
                  "0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22)",
              }}
            >
              <div>
                <div className="">{slide.title2}</div>
                <img
                  src={slide.image2}
                  alt={`Slide ${index + 1}`}
                  className="w-[400px] h-[225px] object-cover"
                />
              </div>
            </div>

            {/* </div> */}
          </SwiperSlide>
        ))}

        {slides.map((slide, index) => (
          <SwiperSlide
            key={index}
            className="lg:hidden flex flex-col lg:flex-row items-center justify-center gap-3 "
          >
            {/* <div className=""> */}
            <div
              className="w-[420px] lg:w-[560px] h-[330px] border border-black bg-white flex xl:gap-2 justify-center items-center"
              style={{
                boxShadow:
                  "0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22)",
              }}
            >
              <div>
                <div className="">{slide.title2}</div>
                <img
                  src={slide.image2}
                  alt={`Slide ${index + 1}`}
                  className="w-[400px] h-[225px] object-cover"
                />
              </div>
            </div>
            <div
              className="hidden w-[420px] lg:w-[560px] h-[330px] border border-black bg-white lg:flex xl:gap-2 justify-center items-center"
              style={{
                boxShadow:
                  "0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22)",
              }}
            >
              <div>
                <div className="">{slide.title2}</div>
                <img
                  src={slide.image2}
                  alt={`Slide ${index + 1}`}
                  className="w-[400px] h-[225px] object-cover"
                />
              </div>
            </div>

            {/* </div> */}
          </SwiperSlide>
        ))}
      </Swiper>
    </div>
  );
};

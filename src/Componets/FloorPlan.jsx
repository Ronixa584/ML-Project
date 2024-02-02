import React from "react";
import { useState } from "react";
import RF1 from "./RF1";

const FloorPlan = () => {
  const [ModalOpen, setModalOpen] = useState(false);

  const openModal = () => {
    setModalOpen(true);
  };

  const closeModal = () => {
    setModalOpen(false);
  };

  return (
    <section id="FloorPlanID" className="pt-[40px]">
      {/* Title Location */}
      <div className="px-6 ">
        <div className="flex justify-center mt-[20px] mb-[40px] text-[27px] font-bold">
          Floor Plan
        </div>
      </div>

      {/* map */}
      <div className="flex items-center justify-center px-6 ">
        <div className="row relative">
          <img
            src="https://dpgvclkvnnsee.cloudfront.net/microsites/woods-platina/lib/images/map.jpg"
            width="100%"
            className="lg:w-[540px] lg:h-[500px] filter blur-sm"
          />
          <button
            onClick={openModal}
            className="bg-black text-white font-medium px-4 py-2 rounded absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"
          >
            View Floor Plan
          </button>
        </div>
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
    </section>
  );
};

export default FloorPlan;

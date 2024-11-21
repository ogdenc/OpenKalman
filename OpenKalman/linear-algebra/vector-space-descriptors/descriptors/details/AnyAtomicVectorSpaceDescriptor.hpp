/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of the AnyAtomicVectorSpaceDescriptor class.
 */

#ifndef OPENKALMAN_ANYATOMICVECTORTYPES_HPP
#define OPENKALMAN_ANYATOMICVECTORTYPES_HPP

#include<variant>
#include<complex>
#include<typeindex>


namespace OpenKalman::descriptors::detail
{
  /**
   * \internal
   * \brief A type representing any \ref atomic_static_vector_space_descriptor object associated with a DynamicDescriptor.
   * \tparam Scalar The scalar type for elements associated with this \ref vector_space_descriptor object.
   */
#ifdef __cpp_concepts
  template<value::number Scalar>
#else
  template<typename...AllowableScalarTypes>
#endif
  struct AnyAtomicVectorSpaceDescriptor
  {
  private:

    using Getter = std::function<Scalar(std::size_t)>;
    using Setter = std::function<void(const Scalar&, std::size_t)>;

    struct Concept
    {
      [[nodiscard]] virtual std::type_index get_type_index() const = 0;
      [[nodiscard]] virtual bool is_euclidean() const = 0;
      [[nodiscard]] virtual std::size_t size() const = 0;
      [[nodiscard]] virtual std::size_t euclidean_size() const = 0;
      [[nodiscard]] virtual Scalar to_euclidean_element(const Getter& g, std::size_t euclidean_local_index, std::size_t start) const = 0;
      [[nodiscard]] virtual Scalar from_euclidean_element(const Getter& g, std::size_t local_index, std::size_t euclidean_start) const = 0;
      [[nodiscard]] virtual Scalar get_wrapped_component(const Getter& g, std::size_t local_index, std::size_t start) const = 0;
      virtual void set_wrapped_component(const Setter& s, const Getter& g, const Scalar& x, std::size_t local_index, std::size_t start) const = 0;
    };


    template <typename T>
    struct Model : Concept
    {
      static Model* get_instance() { static Model instance; return &instance; }

      [[nodiscard]] std::type_index get_type_index() const final { return typeid(T); }

      [[nodiscard]] bool is_euclidean() const final { return euclidean_vector_space_descriptor<T>; }

      [[nodiscard]] std::size_t size() const final { return dimension_size_of_v<T>; }

      [[nodiscard]] std::size_t euclidean_size() const final { return euclidean_dimension_size_of_v<T>; }

      [[nodiscard]] Scalar to_euclidean_element(const Getter& g, std::size_t euclidean_local_index, std::size_t start) const final
      {
        return interface::static_vector_space_descriptor_traits<T>::to_euclidean_element(g, euclidean_local_index, start);
      }

      [[nodiscard]] Scalar from_euclidean_element(const Getter& g, std::size_t local_index, std::size_t euclidean_start) const final
      {
        return interface::static_vector_space_descriptor_traits<T>::from_euclidean_element(g, local_index, euclidean_start);
      }

      [[nodiscard]] Scalar get_wrapped_component(const Getter& g, std::size_t local_index, std::size_t start) const final
      {
        return interface::static_vector_space_descriptor_traits<T>::get_wrapped_component(g, local_index, start);
      }

      void set_wrapped_component(const Setter& s, const Getter& g, const Scalar& x, std::size_t local_index, std::size_t start) const final
      {
        interface::static_vector_space_descriptor_traits<T>::set_wrapped_component(s, g, x, local_index, start);
      }

    };

  public:

    /**
     * \brief Construct from a \ref static_vector_space_descriptor.
     */
#ifdef __cpp_concepts
    template <static_vector_space_descriptor T>
#else
    template<typename T, std::enable_if_t<static_vector_space_descriptor<T>, int> = 0>
#endif
    explicit constexpr AnyAtomicVectorSpaceDescriptor(T&&) : mConcept {Model<T>::get_instance()} {}


    /**
     * \brief Get the std::type_index for the underlying atomic index descritpr.
     */
    [[nodiscard]] std::type_index get_type_index() const { return mConcept->get_type_index(); }


    /**
     * \brief Whether this \ref vector_space_descriptor object is untyped
     * \sa euclidean_vector_space_descriptor
     */
    [[nodiscard]] bool is_euclidean() const { return mConcept->is_euclidean(); }


    [[nodiscard]] std::size_t size() const { return mConcept->size(); }


    [[nodiscard]] std::size_t euclidean_size() const { return mConcept->euclidean_size(); }


    /**
     * \brief Maps an element to coordinates in Euclidean space.
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param euclidean_local_index A local index relative to the Euclidean-transformed coordinates (starting at 0)
     * \param start The starting index within the \ref vector_space_descriptor object
     */
#ifdef __cpp_concepts
    value::number auto
    to_euclidean_element(const std::convertible_to<Getter> auto& g, std::size_t euclidean_local_index, std::size_t start) const
#else
    template<typename G, std::enable_if_t<std::is_convertible_v<G, Getter>, int> = 0>
    auto to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start) const
#endif
    {
      return mConcept->to_euclidean_element(g, euclidean_local_index, start);
    }


    /**
     * \brief Maps a coordinate in Euclidean space to an element.
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index relative to the original coordinates (starting at 0)
     * \param start The starting index within the Euclidean-transformed indices
     */
#ifdef __cpp_concepts
    value::number auto
    from_euclidean_element(const std::convertible_to<Getter> auto& g, std::size_t local_index, std::size_t euclidean_start) const
#else
    template<typename G, std::enable_if_t<is_convertible_v<std::is_convertible_v<G, Getter>, int> = 0>
    auto from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start) const
#endif
    {
      return mConcept->from_euclidean_element(g, local_index, euclidean_start);
    }


    /**
     * \brief Perform modular wrapping of an element.
     * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index relative to the original coordinates (starting at 0)
     * \param start The starting location of the angle within any larger set of \ref vector_space_descriptor
     */
#ifdef __cpp_concepts
    value::number auto
    get_wrapped_component(const std::convertible_to<Getter> auto& g, std::size_t local_index, std::size_t start) const
#else
    template<typename G, std::enable_if_t<is_convertible_v<std::is_convertible_v<G, Getter>, int> = 0>
    auto get_wrapped_component(const G& g, std::size_t local_index, std::size_t start) const
#endif
    {
      return mConcept->get_wrapped_component(g, local_index, start);
    }


    /**
     * \brief Set an element and then perform any necessary modular wrapping.
     * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
     * \param s An element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>)
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param x The scalar value to be set
     * \param local_index A local index relative to the original coordinates (starting at 0)
     * \param start The starting location of the angle within any larger set of \ref vector_space_descriptor
     */
#ifdef __cpp_concepts
    void set_wrapped_component(const std::convertible_to<Setter> auto& s, const std::convertible_to<Getter> auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start) const
#else
    template<typename S, typename G, std::enable_if_t<std::is_convertible_v<S, Setter> and std::is_convertible_v<G, Getter>, int> = 0>
    void set_wrapped_component(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
      std::size_t local_index, std::size_t start) const
#endif
    {
      mConcept->set_wrapped_component(s, g, x, local_index, start);
    }

  private:

    Concept *mConcept;

  };


} // namespace OpenKalman::descriptors::detail


#endif //OPENKALMAN_ANYATOMICVECTORTYPES_HPP

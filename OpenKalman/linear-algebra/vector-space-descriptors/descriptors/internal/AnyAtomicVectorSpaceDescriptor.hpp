/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
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

#include <typeindex>
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \internal
   * \brief A type representing any \ref atomic_static_vector_space_descriptor object associated with a DynamicDescriptor.
   * \tparam Scalar The scalar type for elements associated with this \ref vector_space_descriptor object.
   */
#ifdef __cpp_concepts
  template<value::number Scalar>
#else
  template<typename Scalar>
#endif
  struct AnyAtomicVectorSpaceDescriptor
  {
  private:

    using Getter = std::function<Scalar(std::size_t)>;
    using Setter = std::function<void(const Scalar&, std::size_t)>;

    struct Concept
    {
      virtual ~Concept() = default;
      [[nodiscard]] virtual std::size_t size() const = 0;
      [[nodiscard]] virtual std::size_t euclidean_size() const = 0;
      [[nodiscard]] virtual bool is_euclidean() const = 0;
      [[nodiscard]] virtual std::type_index type_index() const = 0;
      [[nodiscard]] virtual Scalar to_euclidean_element(const Getter& g, std::size_t euclidean_local_index, std::size_t start) const = 0;
      [[nodiscard]] virtual Scalar from_euclidean_element(const Getter& g, std::size_t local_index, std::size_t euclidean_start) const = 0;
      [[nodiscard]] virtual Scalar get_wrapped_component(const Getter& g, std::size_t local_index, std::size_t start) const = 0;
      virtual void set_wrapped_component(const Setter& s, const Getter& g, const Scalar& x, std::size_t local_index, std::size_t start) const = 0;
    };


    template <typename T>
    struct Model : Concept
    {
      static Model* get_instance() { static Model instance; return &instance; }

      [[nodiscard]] std::size_t size() const final { return dimension_size_of_v<T>; }

      [[nodiscard]] std::size_t euclidean_size() const final { return euclidean_dimension_size_of_v<T>; }

      [[nodiscard]] bool is_euclidean() const final { return euclidean_vector_space_descriptor<T>; }

      [[nodiscard]] std::type_index type_index() const final { return typeid(T); }

      [[nodiscard]] Scalar to_euclidean_element(const Getter& g, std::size_t euclidean_local_index, std::size_t start) const final
      {
        return descriptor::to_euclidean_element(T{}, g, euclidean_local_index, start);
      }

      [[nodiscard]] Scalar from_euclidean_element(const Getter& g, std::size_t local_index, std::size_t euclidean_start) const final
      {
        return descriptor::from_euclidean_element(T{}, g, local_index, euclidean_start);
      }

      [[nodiscard]] Scalar get_wrapped_component(const Getter& g, std::size_t local_index, std::size_t start) const final
      {
        return descriptor::get_wrapped_component(T{}, g, local_index, start);
      }

      void set_wrapped_component(const Setter& s, const Getter& g, const Scalar& x, std::size_t local_index, std::size_t start) const final
      {
        descriptor::set_wrapped_component(T{}, s, g, x, local_index, start);
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
    explicit constexpr
    AnyAtomicVectorSpaceDescriptor(const T& t) : mConcept {Model<T>::get_instance()} {}

  private:

    Concept *mConcept;

#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename>
#endif
    friend struct interface::vector_space_traits;

  };


} // namespace OpenKalman::descriptor::internal


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for AnyAtomicVectorSpaceDescriptor.
   */
  template<typename Scalar>
  struct vector_space_traits<descriptor::internal::AnyAtomicVectorSpaceDescriptor<Scalar>>
  {
  private:

    using T = descriptor::internal::AnyAtomicVectorSpaceDescriptor<Scalar>;
    using Getter = std::function<Scalar(std::size_t)>;
    using Setter = std::function<void(const Scalar&, std::size_t)>;

  public:

    using scalar_type = Scalar;


    static constexpr auto
    size(const T& t) { return t.mConcept->size(); }


    static constexpr auto
    euclidean_size(const T& t) { return t.mConcept->euclidean_size(); }


    static constexpr auto
    collection(const T& t) { return std::array {t}; }


    static constexpr auto
    is_euclidean(const T& t) { return t.mConcept->is_euclidean(); }


    static constexpr auto
    type_index(const T& t) { return t.mConcept->type_index(); }


#ifdef __cpp_concepts
    static constexpr value::value auto
    to_euclidean_component(const T& t, const auto& g, const value::index auto& euclidean_local_index, const value::index auto& start)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<Scalar>; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, Scalar> and value::index<L> and value::index<S>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T& t, const Getter& g, const L& euclidean_local_index, const S& start)
#endif
    {
      return t.mConcept->to_euclidean_element(g, euclidean_local_index, start);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    from_euclidean_component(const T& t, const auto& g, const value::index auto& local_index, const value::index auto& euclidean_start)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<Scalar>; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, Scalar>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T& t, const Getter& g, const L& local_index, const S& euclidean_start)
#endif
    {
      return t.mConcept->from_euclidean_element(g, local_index, euclidean_start);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    get_wrapped_component(const T& t, const auto& g, const value::index auto& local_index, const value::index auto& start)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<Scalar>; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, Scalar>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T& t, const Getter& g, const L& local_index, const S& start)
#endif
    {
      return t.mConcept->get_wrapped_component(g, local_index, start);
    }


#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T& t, const auto& s, const auto& g, const Scalar& x,
      const value::index auto& local_index, const value::index auto& start)
    requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
    template<typename Setter, typename Getter, typename L, typename S, std::enable_if_t<
      value::index<L> and value::index<S> and
      std::is_invocable<const Setter&, const Scalar&, std::size_t>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T& t, const Setter& s, const Getter& g, const Scalar& x, const L& local_index, const S& start)
#endif
    {
      t.mConcept->set_wrapped_component(s, g, x, local_index, start);
    }

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_ANYATOMICVECTORTYPES_HPP

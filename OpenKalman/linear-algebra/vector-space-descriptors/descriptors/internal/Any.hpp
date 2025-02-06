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
 * \brief Definition of the descriptor::internal::Any class.
 */

#ifndef OPENKALMAN_ANYATOMICVECTORTYPES_HPP
#define OPENKALMAN_ANYATOMICVECTORTYPES_HPP

#include <typeindex>
#include <memory>
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \internal
   * \brief A type representing any \ref atomic_vector_space_descriptor object associated with a DynamicDescriptor.
   * \tparam Scalar The scalar type for elements associated with this \ref vector_space_descriptor object.
   */
#ifdef __cpp_concepts
  template<value::number Scalar>
#else
  template<typename Scalar>
#endif
  struct Any
  {
  private:

    using Getter = std::function<Scalar(std::size_t)>;
    using Setter = std::function<void(const Scalar&, std::size_t)>;

    struct Base
    {
      virtual ~Base() = default;
      [[nodiscard]] virtual std::size_t size() const = 0;
      [[nodiscard]] virtual std::size_t euclidean_size() const = 0;
      [[nodiscard]] virtual bool is_euclidean() const = 0;
      [[nodiscard]] virtual std::type_index type_index() const = 0;
      [[nodiscard]] virtual Scalar to_euclidean_element(const Getter& g, std::size_t euclidean_local_index) const = 0;
      [[nodiscard]] virtual Scalar from_euclidean_element(const Getter& g, std::size_t local_index) const = 0;
      [[nodiscard]] virtual Scalar get_wrapped_component(const Getter& g, std::size_t local_index) const = 0;
      virtual void set_wrapped_component(const Setter& s, const Getter& g, const Scalar& x, std::size_t local_index) const = 0;
    };


    template <typename T>
    struct Derived : Base
    {
      template<typename Arg>
      explicit Derived(Arg&& arg) : my_t(std::forward<Arg>(arg)) {}

      [[nodiscard]] std::size_t size() const final { return get_size(my_t); }

      [[nodiscard]] std::size_t euclidean_size() const final { return get_euclidean_size(my_t); }

      [[nodiscard]] bool is_euclidean() const final { return get_is_euclidean(my_t); }

      [[nodiscard]] std::type_index type_index() const final { return typeid(T); }

      [[nodiscard]] Scalar to_euclidean_element(const Getter& g, std::size_t euclidean_local_index) const final
      {
        return descriptor::to_euclidean_element(my_t, g, euclidean_local_index);
      }

      [[nodiscard]] Scalar from_euclidean_element(const Getter& g, std::size_t local_index) const final
      {
        return descriptor::from_euclidean_element(my_t, g, local_index);
      }

      [[nodiscard]] Scalar get_wrapped_component(const Getter& g, std::size_t local_index) const final
      {
        return descriptor::get_wrapped_component(my_t, g, local_index);
      }

      void set_wrapped_component(const Setter& s, const Getter& g, const Scalar& x, std::size_t local_index) const final
      {
        descriptor::set_wrapped_component(my_t, s, g, x, local_index);
      }

    private:

      T my_t;
    };

  public:

    /**
     * \brief Construct from a \ref vector_space_descriptor.
     */
#ifdef __cpp_concepts
    template <vector_space_descriptor Arg>
#else
    template<typename Arg, std::enable_if_t<vector_space_descriptor<Arg>, int> = 0>
#endif
    explicit constexpr
    Any(Arg&& arg) : mBase {std::make_shared<Derived<std::decay_t<Arg>>>(std::forward<Arg>(arg))} {}

  private:

    const std::shared_ptr<Base> mBase;

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
   * \brief traits for descriptor::internal::Any.
   */
  template<typename Scalar>
  struct vector_space_traits<descriptor::internal::Any<Scalar>>
  {
  private:

    using T = descriptor::internal::Any<Scalar>;
    using Getter = std::function<Scalar(std::size_t)>;
    using Setter = std::function<void(const Scalar&, std::size_t)>;

  public:

    static constexpr bool is_specialized = true;


    using scalar_type = Scalar;


    static constexpr auto
    size(const T& t) { return t.mBase->size(); }


    static constexpr auto
    euclidean_size(const T& t) { return t.mBase->euclidean_size(); }


    static constexpr auto
    is_euclidean(const T& t) { return t.mBase->is_euclidean(); }


    static constexpr auto
    type_index(const T& t) { return t.mBase->type_index(); }


    static constexpr auto
    component_collection(const T& t) { return std::array {t}; }


#ifdef __cpp_concepts
    static constexpr value::value auto
    to_euclidean_component(const T& t, const auto& g, const value::index auto& euclidean_local_index)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<scalar_type>; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, scalar_type> and value::index<L>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T& t, const Getter& g, const L& euclidean_local_index)
#endif
    {
      return t.mBase->to_euclidean_element(g, euclidean_local_index);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    from_euclidean_component(const T& t, const auto& g, const value::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<scalar_type>; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, scalar_type>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      return t.mBase->from_euclidean_element(g, local_index);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    get_wrapped_component(const T& t, const auto& g, const value::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<scalar_type>; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, scalar_type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      return t.mBase->get_wrapped_component(g, local_index);
    }


#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T& t, const auto& s, const auto& g, const scalar_type& x, const value::index auto& local_index)
    requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
    template<typename Setter, typename Getter, typename L, std::enable_if_t<value::index<L> and
      std::is_invocable<const Setter&, const scalar_type&, std::size_t>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T& t, const Setter& s, const Getter& g, const scalar_type& x, const L& local_index)
#endif
    {
      t.mBase->set_wrapped_component(s, g, x, local_index);
    }

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_ANYATOMICVECTORTYPES_HPP

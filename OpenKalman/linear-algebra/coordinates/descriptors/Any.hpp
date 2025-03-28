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
 * \brief Definition of the coordinate::Any class.
 */

#ifndef OPENKALMAN_ANYATOMICVECTORTYPES_HPP
#define OPENKALMAN_ANYATOMICVECTORTYPES_HPP

#include <memory>
#include "values/concepts/number.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/functions/to_euclidean_element.hpp"
#include "linear-algebra/coordinates/functions/from_euclidean_element.hpp"
#include "linear-algebra/coordinates/functions/get_wrapped_component.hpp"
#include "linear-algebra/coordinates/functions/set_wrapped_component.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \internal
   * \brief A type representing any \ref coordinate::descriptor object associated with a DynamicDescriptor.
   * \tparam Scalar The scalar type for elements associated with this \ref coordinate::pattern object.
   */
#ifdef __cpp_concepts
  template<value::number Scalar = double>
#else
  template<typename Scalar = double>
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
      [[nodiscard]] virtual std::size_t hash_code() const = 0;
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

      [[nodiscard]] std::size_t hash_code() const final { return internal::get_hash_code(my_t); }

      [[nodiscard]] Scalar to_euclidean_element(const Getter& g, std::size_t euclidean_local_index) const final
      {
        return coordinate::to_euclidean_element(my_t, g, euclidean_local_index);
      }

      [[nodiscard]] Scalar from_euclidean_element(const Getter& g, std::size_t local_index) const final
      {
        return coordinate::from_euclidean_element(my_t, g, local_index);
      }

      [[nodiscard]] Scalar get_wrapped_component(const Getter& g, std::size_t local_index) const final
      {
        return coordinate::get_wrapped_component(my_t, g, local_index);
      }

      void set_wrapped_component(const Setter& s, const Getter& g, const Scalar& x, std::size_t local_index) const final
      {
        coordinate::set_wrapped_component(my_t, s, g, x, local_index);
      }

    private:

      T my_t;
    };

  public:

    /**
     * \brief Construct from a \ref coordinate::descriptor.
     */
#ifdef __cpp_concepts
    template <descriptor Arg>
#else
    template<typename Arg, std::enable_if_t<descriptor<Arg>, int> = 0>
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
    friend struct interface::coordinate_descriptor_traits;

  };


} // namespace OpenKalman::coordinate


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for coordinate::Any.
   */
  template<typename Scalar>
  struct coordinate_descriptor_traits<coordinate::Any<Scalar>>
  {
  private:

    using T = coordinate::Any<Scalar>;
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


    static constexpr std::size_t
    hash_code(const T& t) { return t.mBase->hash_code(); }


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


namespace std
{
#ifdef __cpp_concepts
  template<typename Scalar1, std::common_with<Scalar1> Scalar2>
#else
  template<typename Scalar1, typename Scalar2>
#endif
  struct common_type<OpenKalman::coordinate::Any<Scalar1>, OpenKalman::coordinate::Any<Scalar2>>
  {
    using type = OpenKalman::coordinate::Any<common_type_t<Scalar1, Scalar2>>;
  };


#ifdef __cpp_concepts
  template<typename Scalar, OpenKalman::coordinate::descriptor U>
#else
  template<typename Scalar, typename U>
#endif
  struct common_type<OpenKalman::coordinate::Any<Scalar>, U>
  {
    using type = OpenKalman::coordinate::Any<Scalar>;
  };


#ifdef __cpp_concepts
  template<OpenKalman::coordinate::descriptor T, typename Scalar>
#else
  template<typename T, typename Scalar>
#endif
  struct common_type<T, OpenKalman::coordinate::Any<Scalar>>
  {
    using type = OpenKalman::coordinate::Any<Scalar>;
  };
}

#endif //OPENKALMAN_ANYATOMICVECTORTYPES_HPP

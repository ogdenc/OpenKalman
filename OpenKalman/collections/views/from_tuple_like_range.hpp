/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref collections::from_tuple_like_range.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_FROM_TUPLE_LIKE_RANGE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_FROM_TUPLE_LIKE_RANGE_HPP

#include <type_traits>
#include "values/values.hpp"
#include "collections/functions/get.hpp"
#include "collections/traits/size_of.hpp"
#include "internal/movable_wrapper.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A \ref collection_view created from a \ref viewable_tuple_like std::ranges::random_access_range, such as std::array.
   */
#ifdef __cpp_concepts
  template<stdcompat::ranges::random_access_range V> requires
    (stdcompat::ranges::view<stdcompat::remove_cvref_t<V>> or stdcompat::ranges::viewable_range<V>) and
    viewable_tuple_like<V>
#else
  template<typename V>
#endif
  struct from_tuple_like_range : stdcompat::ranges::view_interface<from_tuple_like_range<V>>
  {
  private:

    using RangeBox = internal::movable_wrapper<V>;

  public:

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr
    from_tuple_like_range() = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::default_initializable<RangeBox>, int> = 0>
    constexpr
    from_tuple_like_range() {}
#endif


    /**
     * \brief Construct from a \ref std::ranges::random_access_range.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::same_as<stdcompat::remove_cvref_t<Arg>, from_tuple_like_range>)
#else
    template<typename Arg, std::enable_if_t<(not std::is_same_v<stdcompat::remove_cvref_t<Arg>, from_tuple_like_range>), int> = 0>
#endif
    constexpr explicit 
    from_tuple_like_range(Arg&& arg) noexcept : r_ {std::forward<Arg>(arg)} {}


    /**
     * \brief The base object.
     **/
#ifdef __cpp_explicit_this_parameter
    constexpr decltype(auto)
    base(this auto&& self) noexcept { return std::forward<decltype(self)>(self).r_.get(); }
#else
    constexpr V& base() & { return this->r_.get(); }
    constexpr const V& base() const & { return this->r_.get(); }
    constexpr V&& base() && noexcept { return std::move(*this).r_.get(); }
    constexpr const V&& base() const && noexcept { return std::move(*this).r_.get(); }
#endif


    /**
     * \brief An iterator to the beginning of the range.
     */
    constexpr auto begin()
    {
      return stdcompat::ranges::begin(base());
    }

    /// \overload
    constexpr auto begin() const
    {
      return stdcompat::ranges::begin(base());
    }


    /**
     * \brief An iterator to the end of the range.
     */
    constexpr auto end()
    {
      return stdcompat::ranges::end(base());
    }

    /// \overload
    constexpr auto end() const
    {
      return stdcompat::ranges::end(base());
    }


    /**
     * \brief The size of the resulting object.
     */
    static constexpr auto
    size() { return size_of<V> {}; }


    /**
     * \returns Indicates whether the base object is empty.
     */
    static constexpr auto
    empty()
    {
      if constexpr (size_of_v<V> == 0) return std::true_type{};
      else return std::false_type{};
    }


    /**
     * \returns Returns the initial element.
     */
    constexpr decltype(auto)
    front() { return OpenKalman::internal::generalized_std_get<0>(base()); }

    /// \overload
    constexpr decltype(auto)
    front() const { return OpenKalman::internal::generalized_std_get<0>(base()); }


    /**
     * \returns Returns the final element.
     */
    constexpr decltype(auto)
    back() { return OpenKalman::internal::generalized_std_get<size() - 1_uz>(base()); }

    /// \overload
    constexpr decltype(auto)
    back() const { return OpenKalman::internal::generalized_std_get<size() - 1_uz>(base()); }

    
    /**
     * \brief Subscript operator
     * \note Performs no runtime bounds checking
     */
#ifdef __cpp_explicit_this_parameter
    template<typename Self, values::index I>
    constexpr decltype(auto)
    operator[](this Self&& self, I i) noexcept
    {
      if constexpr (values::fixed<I>) static_assert(values::fixed_value_of<I>::value < size(), "Index out of range");
      return collections::get(std::forward<Self>(self), std::move(i));
    }
#else
    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) &
    {
      if constexpr (values::fixed<I>) static_assert(values::fixed_value_of<I>::value < size(), "Index out of range");
      return collections::get(*this, std::move(i));
    }

    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) const &
    {
      if constexpr (values::fixed<I>) static_assert(values::fixed_value_of<I>::value < size(), "Index out of range");
      return collections::get(*this, std::move(i));
    }

    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) && noexcept
    {
      if constexpr (values::fixed<I>) static_assert(values::fixed_value_of<I>::value < size(), "Index out of range");
      return collections::get(std::move(*this), std::move(i));
    }

    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) const && noexcept
    {
      if constexpr (values::fixed<I>) static_assert(values::fixed_value_of<I>::value < size(), "Index out of range");
      return collections::get(std::move(*this), std::move(i));
    }
#endif


    /**
     * \brief Get element i.
     * \note Performs no runtime bounds checking
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      static_assert(i < size_of_v<V>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(std::forward<decltype(self)>(self).base());
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      static_assert(i < size_of_v<V>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(base());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      static_assert(i < size_of_v<V>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(base());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      static_assert(i < size_of_v<V>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(std::move(*this).base());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      static_assert(i < size_of_v<V>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(std::move(*this).base());
    }
#endif

  private:

    RangeBox r_;

  };


  template<typename V>
  from_tuple_like_range(V&&) -> from_tuple_like_range<V>;

}


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::stdcompat::ranges
#endif
{
  template<typename V>
  constexpr bool enable_borrowed_range<OpenKalman::collections::from_tuple_like_range<V>> =
    std::is_lvalue_reference_v<V> or enable_borrowed_range<OpenKalman::stdcompat::remove_cvref_t<V>>;
}


namespace std
{
  template<typename Tup>
  struct tuple_size<OpenKalman::collections::from_tuple_like_range<Tup>>
    : OpenKalman::collections::size_of<Tup> {};


  template<std::size_t i, typename Tup>
  struct tuple_element<i, OpenKalman::collections::from_tuple_like_range<Tup>>
    : OpenKalman::collections::collection_element<i, Tup> {};
}


#endif 

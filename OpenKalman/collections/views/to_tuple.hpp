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
 * \brief Definition for \ref collections::to_tuple.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_TO_TUPLE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_TO_TUPLE_HPP

#include <type_traits>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/language-features.hpp"
#include "basics/compatibility/ranges.hpp"
#include "basics/compatibility/views/view_interface.hpp"
#endif
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "collections/functions/get_size.hpp"
#include "collections/functions/get.hpp"
#include "collections/functions/compare.hpp"
#include "internal/movable_wrapper.hpp"

namespace OpenKalman::collections
{
  namespace detail_to_tuple
  {
    using namespace std;
#ifdef __cpp_lib_ranges
    namespace ranges = std::ranges;
#endif


#ifdef __cpp_concepts
    template<std::size_t i, typename T>
#else
    template<std::size_t i, typename T, typename = void>
#endif
    struct tuple_element_impl
    {
      using type = ranges::range_value_t<T>;

      template<typename R> constexpr decltype(auto) operator() (R&& r) const
       { return collections::get(std::forward<R>(r), std::integral_constant<std::size_t, i>{}); }
    };


#ifdef __cpp_concepts
    template<std::size_t i, typename T> requires gettable<i, T>
    struct tuple_element_impl<i, T>
#else
    template<std::size_t i, typename T>
    struct tuple_element_impl<i, T, std::enable_if_t<gettable<i, T>>>
#endif
      : std::tuple_element<i, T>
    {
      using type = std::tuple_element_t<i, T>;

      template<typename R> constexpr decltype(auto) operator() (R&& r) const
       { return OpenKalman::internal::generalized_std_get<i>(std::forward<R>(r)); }
    };


#ifdef __cpp_concepts
    template<std::size_t i, typename T>
#else
    template<std::size_t i, typename T, typename = void>
#endif
    struct get_elem : tuple_element_impl<i, T> {};


    template<std::size_t i, typename R>
    inline constexpr decltype(auto)
    get_from_base(R&& r) { using namespace std; return get_elem<i, remove_cvref_t<R>>{}(std::forward<R>(r)); }


    template<typename T>
    struct get_elem<0, ranges::single_view<T>>
    {
      using type = T;
      template<typename U> constexpr decltype(auto) operator() (U&& u) const { return *std::forward<U>(u).data(); }
    };


    template<std::size_t i, typename R>
    struct get_elem<i, ranges::ref_view<R>> : get_elem<i, remove_cvref_t<R>>
    {
      template<typename T> constexpr decltype(auto) operator() (T&& t) const { return get_from_base<i>(std::forward<T>(t).base()); }
    };


    template<std::size_t i, typename R>
    struct get_elem<i, ranges::owning_view<R>> : get_elem<i, remove_cvref_t<R>>
    {
      template<typename T> constexpr decltype(auto) operator() (T&& t) const { return get_from_base<i>(std::forward<T>(t).base()); }
    };


#ifdef __cpp_concepts
    template<std::size_t i, typename V> requires (size_of_v<V> != dynamic_size)
    struct get_elem<i, ranges::reverse_view<V>>
#else
    template<std::size_t i, typename V>
    struct get_elem<i, ranges::reverse_view<V>, std::enable_if_t<size_of_v<V> != dynamic_size>>
#endif
      : get_elem<size_of_v<V> - i - 1_uz, remove_cvref_t<V>>
    {
      template<typename T> constexpr decltype(auto) operator() (T&& t) const
      {
        if constexpr (not std::is_lvalue_reference_v<T> or std::is_copy_constructible_v<V>)
          return get_from_base<size_of_v<V> - i - 1_uz>(std::forward<T>(t).base());
        else
          return collections::get(std::forward<T>(t), std::integral_constant<std::size_t, i>{});
      }
    };

  }


  /**
   * \brief A \ref collection_view created from a std::ranges::random_access_range that is a std::ranges::viewable_range.
   * \details Presumably, V is not \ref uniform_tuple_like, but may have some tuple-like properties such as a defined std::tuple_size.
   */
#ifdef __cpp_lib_ranges
  template<std::ranges::random_access_range V> requires std::ranges::view<std::remove_cvref_t<V>> or std::ranges::viewable_range<V>
  struct to_tuple : std::ranges::view_interface<to_tuple<V>>
#else
  template<typename V>
  struct to_tuple : ranges::view_interface<to_tuple<V>>
#endif
  {
  private:

    using RangeBox = internal::movable_wrapper<V>;

  public:

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr
    to_tuple() = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and std::is_default_constructible_v<RangeBox>, int> = 0>
    constexpr
    to_tuple() {}
#endif


    /**
     * \brief Construct from a \ref std::ranges::random_access_range.
     */
#if defined(__cpp_concepts) and defined(__cpp_lib_remove_cvref)
    template<typename Arg> requires std::constructible_from<RangeBox, Arg&&> and (not std::same_as<std::remove_cvref_t<Arg>, to_tuple>)
#else
    template<typename Arg, std::enable_if_t<
      std::is_constructible_v<RangeBox, Arg&&> and (not std::is_same_v<remove_cvref_t<Arg>, to_tuple>), int> = 0>
#endif
    constexpr explicit 
    to_tuple(Arg&& arg) noexcept : r_ {std::forward<Arg>(arg)} {}


    /**
     * \brief The base tuple.
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
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#endif
      return ranges::begin(base());
    }

    /// \overload
    constexpr auto begin() const
    {
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#endif
      return ranges::begin(base());
    }


    /**
     * \brief An iterator to the end of the range.
     */
    constexpr auto end()
    {
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#endif
      return ranges::end(base());
    }

    /// \overload
    constexpr auto end() const
    {
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#endif
      return ranges::end(base());
    }


    /**
     * \brief The size of the resulting object.
     */
#ifdef __cpp_concepts
    constexpr auto
    size() const requires sized<V>
#else
    template<bool Enable = true, std::enable_if_t<Enable and sized<V>, int> = 0>
    constexpr auto
    size() const
#endif
    {
      return get_size(base());
    }


    /**
     * \returns Indicates whether the base object is empty.
     */
#ifdef __cpp_concepts
    constexpr auto
    empty() const requires sized<V> or std::ranges::forward_range<V>
#else
    template<bool Enable = true, std::enable_if_t<Enable and (sized<V> or ranges::forward_range<V>), int> = 0>
    constexpr auto
    empty() const
#endif
    {
      if constexpr (sized<V>)
      {
        if constexpr (size_of_v<V> == dynamic_size) return get_size(base()) == 0_uz;
        else if constexpr (size_of_v<V> == 0) return std::true_type{};
        else return std::false_type{};
      }
      else
      {
#ifdef __cpp_lib_ranges
        namespace ranges = std::ranges;
#endif
        return ranges::begin(base()) == ranges::end(base());
      }
    }


    /**
     * \returns Returns the initial element.
     */
    constexpr decltype(auto)
    front() { return collections::get(base(), std::integral_constant<std::size_t, 0>{}); }

    /// \overload
    constexpr decltype(auto)
    front() const { return collections::get(base(), std::integral_constant<std::size_t, 0>{}); }

  private:

#ifdef __cpp_lib_ranges
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct back_gettable : std::false_type {};

#ifdef __cpp_lib_ranges
    template<sized T>
    struct back_gettable<T>
#else
    template<typename T>
    struct back_gettable<T, std::enable_if_t<sized<T>>>
#endif
      : std::bool_constant<size_of_v<T> != dynamic_size and gettable<size_of_v<T> - 1_uz, T>> {};

  public:

    /**
     * \returns Returns the final element.
     */
#ifdef __cpp_lib_ranges
    constexpr decltype(auto)
    back() requires (std::ranges::bidirectional_range<V> and std::ranges::common_range<V>) or back_gettable<V>::value
#else
    template<bool Enable = true, std::enable_if_t<Enable and
      (ranges::bidirectional_range<V> and ranges::common_range<V>) or back_gettable<V>::value, int> = 0>
    constexpr decltype(auto)
    back()
#endif
    {
      if constexpr (back_gettable<V>::value)
      { return OpenKalman::internal::generalized_std_get<size_of_v<V> - 1_uz>(base()); }
      else
      {
#ifdef __cpp_lib_ranges
        namespace ranges = std::ranges;
#endif
        using namespace std; return ranges::view_interface<to_tuple<V>>::back();
      }
    }

    /// \overload
#ifdef __cpp_lib_ranges
    constexpr decltype(auto)
    back() const requires (std::ranges::bidirectional_range<V> and std::ranges::common_range<V>) or back_gettable<V>::value
#else
    template<bool Enable = true, std::enable_if_t<Enable and
      (ranges::bidirectional_range<V> and ranges::common_range<V>) or back_gettable<V>::value, int> = 0>
    constexpr decltype(auto)
    back() const
#endif
    {
      if constexpr (back_gettable<V>::value)
      { return OpenKalman::internal::generalized_std_get<size_of_v<V> - 1_uz>(base()); }
      else
      {
#ifdef __cpp_lib_ranges
        namespace ranges = std::ranges;
#endif
        using namespace std; return ranges::view_interface<to_tuple<V>>::back();
      }
    }

    
    /**
     * \brief Subscript operator
     */
#ifdef __cpp_explicit_this_parameter
    template<typename Self, values::index I>
    constexpr decltype(auto)
    operator[](this Self&& self, I i) noexcept
    {
      if constexpr (values::fixed<I> and sized<V>)
        static_assert(size_of_v<V> == dynamic_size or values::fixed_number_of_v<I> < size_of_v<V>, "Index out of range");
      return collections::get(std::forward<Self>(self), std::move(i));
    }
#else
    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) &
    {
      if constexpr (values::fixed<I> and sized<V>)
        static_assert(size_of_v<V> == dynamic_size or values::fixed_number_of_v<I> < size_of_v<V>, "Index out of range");
      return collections::get(*this, std::move(i));
    }

    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) const &
    {
      if constexpr (values::fixed<I> and sized<V>)
        static_assert(size_of_v<V> == dynamic_size or values::fixed_number_of_v<I> < size_of_v<V>, "Index out of range");
      return collections::get(*this, std::move(i));
    }

    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) && noexcept
    {
      if constexpr (values::fixed<I> and sized<V>)
        static_assert(size_of_v<V> == dynamic_size or values::fixed_number_of_v<I> < size_of_v<V>, "Index out of range");
      return collections::get(std::move(*this), std::move(i));
    }

    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) const && noexcept
    {
      if constexpr (values::fixed<I> and sized<V>)
        static_assert(size_of_v<V> == dynamic_size or values::fixed_number_of_v<I> < size_of_v<V>, "Index out of range");
      return collections::get(std::move(*this), std::move(i));
    }
#endif


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      if constexpr(sized<V>) if constexpr (size_of_v<V> != dynamic_size) static_assert(i < size_of_v<V>, "Index out of range");
      return detail_to_tuple::get_from_base<i>(std::forward<decltype(self)>(self).base());
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      if constexpr(sized<V>) if constexpr (size_of_v<V> != dynamic_size) static_assert(i < size_of_v<V>, "Index out of range");
      return detail_to_tuple::get_from_base<i>(base());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      if constexpr(sized<V>) if constexpr (size_of_v<V> != dynamic_size) static_assert(i < size_of_v<V>, "Index out of range");
      return detail_to_tuple::get_from_base<i>(base());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      if constexpr(sized<V>) if constexpr (size_of_v<V> != dynamic_size) static_assert(i < size_of_v<V>, "Index out of range");
      return detail_to_tuple::get_from_base<i>(std::move(*this).base());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      if constexpr(sized<V>) if constexpr (size_of_v<V> != dynamic_size) static_assert(i < size_of_v<V>, "Index out of range");
      return detail_to_tuple::get_from_base<i>(std::move(*this).base());
    }
#endif

  private:

    RangeBox r_;

  };


  template<typename V>
  to_tuple(V&&) -> to_tuple<V>;


#ifndef __cpp_concepts
  namespace detail_to_tuple
  {
    template<typename V, typename = void>
    struct tuple_size {};

    template<typename V>
    struct tuple_size<V, std::enable_if_t<sized<V> and size_of<V>::value != dynamic_size>> : size_of<V> {};
  }
#endif

} // namespace OpenKalman::collections


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::ranges
#endif
{
  template<typename V>
  constexpr bool enable_borrowed_range<OpenKalman::collections::to_tuple<V>> =
    std::is_lvalue_reference_v<V> or enable_borrowed_range<remove_cvref_t<V>>;
}


namespace std
{
#ifdef __cpp_concepts
  template<OpenKalman::collections::sized V> requires (OpenKalman::collections::size_of_v<V> != OpenKalman::dynamic_size)
  struct tuple_size<OpenKalman::collections::to_tuple<V>> : OpenKalman::collections::size_of<V> {};
#else
  template<typename V>
  struct tuple_size<OpenKalman::collections::to_tuple<V>> : OpenKalman::collections::detail_to_tuple::tuple_size<V> {};
#endif


  template<std::size_t i, typename V>
  struct tuple_element<i, OpenKalman::collections::to_tuple<V>>
#ifdef __cpp_lib_remove_cvref
    : OpenKalman::collections::detail_to_tuple::get_elem<i, remove_cvref_t<V>> {};
#else
    : OpenKalman::collections::detail_to_tuple::get_elem<i, OpenKalman::remove_cvref_t<V>> {};
#endif
}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_TO_TUPLE_HPP

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
 * \internal
 * \brief Definition of \ref collections::replicate_view and \ref views::replicate.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_REPLICATE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_REPLICATE_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/ranges.hpp"
#endif
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "values/classes/operation.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/concepts/viewable_collection.hpp"
#include "collections/functions/get_collection_size.hpp"
#include "collections/functions/get.hpp"
#include "internal/maybe_tuple_size.hpp"
#include "internal/maybe_tuple_element.hpp"
#include "collection_view_interface.hpp"
#include "all.hpp"

namespace OpenKalman::collections
{
  namespace internal
  {
    /**
     * \internal
     * \brief Iterator for \ref replicate_view
     */
    template<typename R, typename F>
    struct replicate_view_iterator
    {
      using iterator_category = std::random_access_iterator_tag;
#ifdef __cpp_lib_ranges
      using value_type = std::ranges::range_value_t<R>;
#else
      using value_type = ranges::range_value_t<R>;
#endif
      using difference_type = std::ptrdiff_t;
      explicit constexpr replicate_view_iterator(R& r, F f, std::size_t p) noexcept
        : r_ptr {std::addressof(r)}, factor{std::move(f)}, current{static_cast<difference_type>(p)} {}
      constexpr replicate_view_iterator() = default;
      constexpr replicate_view_iterator(const replicate_view_iterator& other) = default;
      constexpr replicate_view_iterator(replicate_view_iterator&& other) noexcept = default;
      constexpr replicate_view_iterator& operator=(const replicate_view_iterator& other) = default;
      constexpr replicate_view_iterator& operator=(replicate_view_iterator&& other) noexcept = default;
      explicit constexpr operator std::size_t() const noexcept { return static_cast<std::size_t>(current); }
#ifdef __cpp_lib_ranges
      constexpr decltype(auto) operator*() noexcept { return get(*r_ptr, current % std::ranges::size(*r_ptr)); }
      constexpr decltype(auto) operator*() const noexcept { return get(*r_ptr, current % std::ranges::size(*r_ptr)); }
      constexpr decltype(auto) operator[](difference_type offset) noexcept { return get(*r_ptr, (current + offset) % std::ranges::size(*r_ptr)); }
      constexpr decltype(auto) operator[](difference_type offset) const noexcept { return get(*r_ptr, (current + offset) % std::ranges::size(*r_ptr)); }
#else
      constexpr decltype(auto) operator*() noexcept { return get(*r_ptr, current % ranges::size(*r_ptr)); }
      constexpr decltype(auto) operator*() const noexcept { return get(*r_ptr, current % ranges::size(*r_ptr)); }
      constexpr decltype(auto) operator[](difference_type offset) noexcept { return get(*r_ptr, (current + offset) % ranges::size(*r_ptr)); }
      constexpr decltype(auto) operator[](difference_type offset) const noexcept { return get(*r_ptr, (current + offset) % ranges::size(*r_ptr)); }
#endif
      constexpr auto& operator++() noexcept { ++current; return *this; }
      constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() noexcept { --current; return *this; }
      constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) noexcept { current += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) noexcept { current -= diff; return *this; }
      friend constexpr auto operator+(const replicate_view_iterator& it, const difference_type diff) noexcept
      { return replicate_view_iterator {*it.r_ptr, it.factor, static_cast<std::size_t>(it.current + diff)}; }
      friend constexpr auto operator+(const difference_type diff, const replicate_view_iterator& it) noexcept
      { return replicate_view_iterator {*it.r_ptr, it.factor, static_cast<std::size_t>(diff + it.current)}; }
      friend constexpr auto operator-(const replicate_view_iterator& it, const difference_type diff)
      { if (it.current < diff) throw std::out_of_range{"Iterator out of range"};
        return replicate_view_iterator {*it.r_ptr, it.factor, static_cast<std::size_t>(it.current - diff)}; }
      friend constexpr difference_type operator-(const replicate_view_iterator& it, const replicate_view_iterator& other) noexcept
      { return it.current - other.current; }
      friend constexpr bool operator==(const replicate_view_iterator& it, const replicate_view_iterator& other) noexcept
      { return it.current == other.current; }
#ifdef __cpp_impl_three_way_comparison
      constexpr auto operator<=>(const replicate_view_iterator& other) const noexcept { return current <=> other.current; }
#else
      constexpr bool operator!=(const replicate_view_iterator& other) const noexcept { return current != other.current; }
      constexpr bool operator<(const replicate_view_iterator& other) const noexcept { return current < other.current; }
      constexpr bool operator>(const replicate_view_iterator& other) const noexcept { return current > other.current; }
      constexpr bool operator<=(const replicate_view_iterator& other) const noexcept { return current <= other.current; }
      constexpr bool operator>=(const replicate_view_iterator& other) const noexcept { return current >= other.current; }
#endif

    private:

      R* r_ptr; // \todo Convert this to std::shared_ptr<Derived> if p3037Rx ("constexpr std::shared_ptr") is adopted
      F factor;
      difference_type current;

    };


    template<typename R, typename F>
    replicate_view_iterator(R&, const F&, std::size_t) -> replicate_view_iterator<R, F>;

  }


  /**
   * \internal
   * \brief A view that replicates a \ref collection some number of times.
   * \details
   * The following should compile:
   * \code
   * static_assert(std::tuple_size_v<replicate_view<std::tuple<int, double>, std::integral_constant<std::size_t, 3>>> == 6);
   * static_assert(std::is_same_v<std::tuple_element_t<5, replicate_view<std::tuple<double, int, float>, std::integral_constant<std::size_t, 2>>>, float>);
   * static_assert(get<3>(replicate_view {std::tuple{4, 5.}, std::integral_constant<std::size_t, 2>{}}) == 5);
   * static_assert((replicate_view {std::vector{3, 4, 5}, 2u}[4]), 4);
   * static_assert((replicate_view {std::vector{3, 4, 5}, 2u}[std::integral_constant<std::size_t, 5>{}]), 5);
   * \endcode
   * \sa views::replicate
   */
#ifdef __cpp_lib_ranges
  template<collection T, value::index Factor> requires std::same_as<std::decay_t<Factor>, Factor>
#else
  template<typename T, typename Factor>
#endif
  struct replicate_view : collection_view_interface<replicate_view<T, Factor>>
  {
  private:

    using MyT = std::conditional_t<tuple_like<T> and value::dynamic<Factor>,
      all_view<T>, OpenKalman::internal::movable_wrapper<T>>;

    constexpr decltype(auto) get_t() & noexcept
    { if constexpr (tuple_like<T> and value::dynamic<Factor>) return std::forward<MyT&>(my_t); else return my_t.get(); }

    constexpr decltype(auto) get_t() const & noexcept
    { if constexpr (tuple_like<T> and value::dynamic<Factor>) return std::forward<const MyT&>(my_t); else return my_t.get(); }

    constexpr decltype(auto) get_t() && noexcept
    { if constexpr (tuple_like<T> and value::dynamic<Factor>) return std::forward<MyT&&>(my_t); else return std::move(*this).my_t.get(); }

    constexpr decltype(auto) get_t() const && noexcept
    { if constexpr (tuple_like<T> and value::dynamic<Factor>) return std::forward<const MyT&&>(my_t); else return std::move(*this).my_t.get(); }

  public:

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr
    replicate_view() requires std::default_initializable<MyT> and std::default_initializable<Factor> = default;
#else
    template<typename aT = MyT, std::enable_if_t<std::is_default_constructible_v<aT> and std::is_default_constructible_v<Factor>, int> = 0>
    constexpr
    replicate_view() {}
#endif


    /**
     * \brief Construct from a \ref collection.
     */
#ifdef __cpp_concepts
    template<typename Arg, value::index F> requires std::constructible_from<MyT, Arg&&> and std::constructible_from<Factor, F&&>
#else
    template<typename Arg, typename F, std::enable_if_t<value::index<F> and
      std::is_constructible_v<MyT, Arg&&> and std::is_constructible_v<Factor, F&&>, int> = 0>
#endif
    explicit constexpr
    replicate_view(Arg&& arg, F f) : my_t {std::forward<Arg>(arg)}, factor {std::move(f)} {}


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i> requires tuple_like<T>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      if constexpr (size_of_v<T> != dynamic_size and value::fixed<Factor>)
        static_assert(i < size_of_v<T> * value::fixed_number_of_v<Factor>, "Index out of range");
      return collections::get(std::forward<decltype(self)>(self).get_t(),
        value::operation {std::modulus{}, std::integral_constant<std::size_t, i>{}, get_collection_size(self.get_t())});
    }
#else
    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() &
    {
      if constexpr (size_of_v<T> != dynamic_size and value::fixed<Factor>)
        static_assert(i < size_of_v<T> * value::fixed_number_of_v<Factor>, "Index out of range");
      return collections::get(get_t(),
        value::operation {std::modulus{}, std::integral_constant<std::size_t, i>{}, get_collection_size(get_t())});
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() const &
    {
      if constexpr (size_of_v<T> != dynamic_size and value::fixed<Factor>)
        static_assert(i < size_of_v<T> * value::fixed_number_of_v<Factor>, "Index out of range");
      return collections::get(get_t(),
        value::operation {std::modulus{}, std::integral_constant<std::size_t, i>{}, get_collection_size(get_t())});
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() && noexcept
    {
      if constexpr (size_of_v<T> != dynamic_size and value::fixed<Factor>)
        static_assert(i < size_of_v<T> * value::fixed_number_of_v<Factor>, "Index out of range");
      return collections::get(std::move(*this).get_t(),
        value::operation {std::modulus{}, std::integral_constant<std::size_t, i>{}, get_collection_size(std::move(*this).get_t())});
    }

    template<std::size_t i, typename aT = T, std::enable_if_t<tuple_like<aT>, int> = 0>
    constexpr decltype(auto)
    get() const && noexcept
    {
      if constexpr (size_of_v<T> != dynamic_size and value::fixed<Factor>)
        static_assert(i < size_of_v<T> * value::fixed_number_of_v<Factor>, "Index out of range");
      return collections::get(std::move(*this).get_t(),
        value::operation {std::modulus{}, std::integral_constant<std::size_t, i>{}, get_collection_size(std::move(*this).get_t())});
    }
#endif


    /**
     * \returns The size of the object.
     */
#ifdef __cpp_concepts
    constexpr value::index auto size() const
#else
    constexpr auto size() const
#endif
    {
      return value::operation {std::multiplies{}, get_collection_size(get_t()), factor};
    }


    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
#ifdef __cpp_concepts
    constexpr auto
    begin() noexcept requires sized_random_access_range<T> or value::dynamic<Factor>
#else
    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT> or value::dynamic<Factor>, int> = 0>
    constexpr auto
    begin() noexcept
#endif
    {
      return internal::replicate_view_iterator {get_t(), factor, 0};
    }


    /// \overload
#ifdef __cpp_concepts
    constexpr auto
    begin() const noexcept requires sized_random_access_range<T> or value::dynamic<Factor>
#else
    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT> or value::dynamic<Factor>, int> = 0>
    constexpr auto
    begin() const noexcept
#endif
    {
      return internal::replicate_view_iterator {get_t(), factor, 0};
    }


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
#ifdef __cpp_concepts
    constexpr auto
    end() noexcept requires sized_random_access_range<T> or value::dynamic<Factor>
#else
    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT> or value::dynamic<Factor>, int> = 0>
    constexpr auto
    end() noexcept
#endif
    {
      return internal::replicate_view_iterator {get_t(), factor, value::to_number(size())};
    }


    /// \overload
#ifdef __cpp_concepts
    constexpr auto
    end() const noexcept requires sized_random_access_range<T> or value::dynamic<Factor>
#else
    template<typename aT = T, std::enable_if_t<sized_random_access_range<aT> or value::dynamic<Factor>, int> = 0>
    constexpr auto
    end() const noexcept
#endif
    {
      return internal::replicate_view_iterator {get_t(), factor, value::to_number(size())};
    }

  private:

    MyT my_t;
    Factor factor;
  };


  /**
   * \brief Deduction guide
   */
  template<typename Arg, typename F>
  replicate_view(Arg&&, const F&) -> replicate_view<Arg, F>;

} // namespace OpenKalman::collections


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::ranges
#endif
{
  template<typename T, typename F>
  constexpr bool enable_borrowed_range<OpenKalman::collections::replicate_view<T, F>> =
    std::is_lvalue_reference_v<T> or enable_borrowed_range<remove_cvref_t<T>>;
}


namespace std
{
  template<typename T, typename F>
  struct tuple_size<OpenKalman::collections::replicate_view<T, F>>
    : OpenKalman::value::operation<multiplies<size_t>, OpenKalman::collections::internal::maybe_tuple_size<std::decay_t<T>>, F> {};

  template<size_t i, typename T, typename F>
  struct tuple_element<i, OpenKalman::collections::replicate_view<T, F>>
    : OpenKalman::collections::internal::maybe_tuple_element<
        OpenKalman::value::operation<
          std::modulus<std::size_t>,
          integral_constant<std::size_t, i>,
          OpenKalman::collections::internal::maybe_tuple_size<std::decay_t<T>>>, T> {};

  template<typename R, typename F>
  struct iterator_traits<OpenKalman::collections::internal::replicate_view_iterator<R, F>>
  {
    using difference_type = typename OpenKalman::collections::internal::replicate_view_iterator<R, F>::difference_type;
    using value_type = typename OpenKalman::collections::internal::replicate_view_iterator<R, F>::value_type;
    using iterator_category = typename OpenKalman::collections::internal::replicate_view_iterator<R, F>::iterator_category;
  };

} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<value::index Factor>
#else
    template<typename Factor>
#endif
    struct replicate_closure
#if __cpp_lib_ranges >= 202202L
      : std::ranges::range_adaptor_closure<replicate_closure<Factor>>
#endif
    {
      constexpr replicate_closure(Factor f) : factor {std::move(f)} {};

#ifdef __cpp_concepts
      template<viewable_collection R>
#else
      template<typename R, std::enable_if_t<viewable_collection<R>, int> = 0>
#endif
      constexpr auto
      operator() (R&& r) const { return replicate_view {std::forward<R>(r), factor}; }

    private:
      Factor factor;
    };


    struct replicate_adaptor
    {
#ifdef __cpp_concepts
      template<value::index Factor>
#else
      template<typename Factor, std::enable_if_t<value::index<Factor>, int> = 0>
#endif
      constexpr auto
      operator() (Factor factor) const
      {
        return replicate_closure<Factor> {std::move(factor)};
      }


#ifdef __cpp_concepts
      template<viewable_collection R, value::index Factor>
#else
      template<typename R, typename Factor, std::enable_if_t<viewable_collection<R> and value::index<Factor>, int> = 0>
#endif
      constexpr auto
      operator() (R&& r, Factor factor) const
      {
        return replicate_view {std::forward<R>(r), std::move(factor)};
      }

    };

  }


  /**
   * \brief a std::ranges::range_adaptor_closure associated with \ref replicate_view.
   * \details The expression <code>views::replicate{f}(arg)</code> is expression-equivalent
   * to <code>replicate_view(arg, f)</code> for any suitable \ref viewable_collection arg.
   * \sa replicate_view
   */
  inline constexpr detail::replicate_adaptor replicate;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_REPLICATE_HPP

/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for StaticDescriptor class specializations and associated aliases.
 */

#ifndef OPENKALMAN_STATICDESCRIPTOR_HPP
#define OPENKALMAN_STATICDESCRIPTOR_HPP

#include <array>
#include <functional>
#include <typeindex>
#include "basics/utils.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/to_euclidean_element.hpp"
#include "linear-algebra/vector-space-descriptors/functions/from_euclidean_element.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_wrapped_component.hpp"
#include "linear-algebra/vector-space-descriptors/functions/set_wrapped_component.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/euclidean_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/vector_space_component_count.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/maybe_equivalent_to.hpp"
#include "linear-algebra/vector-space-descriptors/functions/comparison-operators.hpp"
#include "internal/AnyAtomicVectorSpaceDescriptor.hpp"

namespace OpenKalman::descriptor
{
#ifdef __cpp_concepts
  template<static_vector_space_descriptor...Cs>
#else
  template<typename...Cs>
#endif
  struct StaticDescriptor
  {
#ifndef __cpp_concepts
    static_assert((static_vector_space_descriptor<Cs> and ...));
#endif


    /**
     * \brief Prepend a set of new \ref vector_space_descriptor to the existing set.
     * \tparam Cnew The set of new coordinates to prepend.
     */
    template<typename ... Cnew>
    using Prepend = StaticDescriptor<Cnew..., Cs ...>;


    /**
     * \brief Append a set of new coordinates to the existing set.
     * \tparam Cnew The set of new coordinates to append.
     */
    template<typename ... Cnew>
    using Append = StaticDescriptor<Cs ..., Cnew ...>;

  private:

    template<std::size_t count, typename...Ds>
    struct Take_impl { using type = StaticDescriptor<>; };


    template<std::size_t count, typename D, typename...Ds>
    struct Take_impl<count, D, Ds...> { using type = typename StaticDescriptor<Ds...>::template Take<count - 1>::template Prepend<D>; };


    template<typename D, typename...Ds>
    struct Take_impl<0, D, Ds...> { using type = StaticDescriptor<>; };

  public:

    /**
     * \brief Take the first <code>count</code> \ref vector_space_descriptor.
     * \tparam count The number of \ref vector_space_descriptor to take.
     */
#ifdef __cpp_concepts
    template<std::size_t count> requires (count <= sizeof...(Cs))
#else
    template<std::size_t count, std::enable_if_t<(count <= sizeof...(Cs)), int> = 0>
#endif
    using Take = typename Take_impl<count, Cs...>::type;


  private:

    template<std::size_t count, typename...Ds>
    struct Drop_impl { using type = StaticDescriptor<>; };


    template<std::size_t count, typename D, typename...Ds>
    struct Drop_impl<count, D, Ds...> { using type = typename StaticDescriptor<Ds...>::template Drop<count - 1>; };


    template<typename D, typename...Ds>
    struct Drop_impl<0, D, Ds...> { using type = StaticDescriptor<D, Ds...>; };

  public:

    /**
     * \brief Drop the first <code>count</code> \ref vector_space_descriptor objects.
     */
#ifdef __cpp_concepts
    template<std::size_t count> requires (count <= sizeof...(Cs))
#else
    template<std::size_t count, std::enable_if_t<(count <= sizeof...(Cs)), int> = 0>
#endif
    using Drop = typename Drop_impl<count, Cs...>::type;

  private:

    template<std::size_t i, typename...Ds>
    struct Select_impl;


    template<std::size_t i, typename D, typename...Ds>
    struct Select_impl<i, D, Ds...> { using type = typename StaticDescriptor<Ds...>::template Select<i - 1>; };


    template<typename D, typename...Ds>
    struct Select_impl<0, D, Ds...> { using type = D; };

  public:

    /**
     * \brief Extract a particular component from the set of fixed \ref vector_space_descriptor.
     * \tparam i The index of the \ref vector_space_descriptor component.
     */
#ifdef __cpp_concepts
    template<std::size_t i> requires (i < sizeof...(Cs))
#else
    template<std::size_t i, std::enable_if_t<(i < sizeof...(Cs)), int> = 0>
#endif
    using Select = typename Select_impl<i, Cs...>::type;


  }; // struct StaticDescriptor

} // namespace OpenKalman::descriptor


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for StaticDescriptor.
   */
  template<typename...Cs>
  struct vector_space_traits<descriptor::StaticDescriptor<Cs...>>
  {
  private:

    using T = descriptor::StaticDescriptor<Cs...>;

  public:

    static constexpr auto
    size(const T&)
    {
      return std::integral_constant<std::size_t, (0 + ... + descriptor::dimension_size_of_v<Cs>)>{};
    }


    static constexpr auto
    euclidean_size(const T&)
    {
      return std::integral_constant<std::size_t, (0 + ... + descriptor::euclidean_dimension_size_of_v<Cs>)>{};
    }

  private:

    template<typename...Ts>
    static constexpr auto combine_euclidean()
    {
      constexpr std::size_t N = (0 + ... + descriptor::dimension_size_of_v<Ts>);
      if constexpr (N == 0) return std::tuple {};
      else return std::tuple {descriptor::Dimensions<N>{}};
    }


    template<typename Arg>
    static constexpr auto canonical_tuple_collection(const Arg& arg)
    {
      constexpr auto s = std::tuple_size_v<Arg>;
      if constexpr (s < 2)
      {
        return arg;
      }
      else
      {
        using A0 = std::tuple_element_t<0, Arg>;
        using A1 = std::tuple_element_t<1, Arg>;
        if constexpr (descriptor::euclidean_vector_space_descriptor<A0> and descriptor::euclidean_vector_space_descriptor<A1>)
        {
          return canonical_tuple_collection(std::tuple_cat(combine_euclidean<A0, A1>(), internal::tuple_slice<2, s>(arg)));
        }
        else if constexpr (descriptor::euclidean_vector_space_descriptor<A0>)
        {
          return std::tuple_cat(
            std::tuple_cat(combine_euclidean<A0>(), std::tuple {std::get<1>(arg)}),
            canonical_tuple_collection(OpenKalman::internal::tuple_slice<2, s>(arg)));
        }
        else
        {
          return std::tuple_cat(
            std::tuple {std::get<0>(arg)},
            canonical_tuple_collection(OpenKalman::internal::tuple_slice<1, s>(arg)));
        }
      }
    }

  public:

    static constexpr auto
    collection(const T& t)
    {
      return canonical_tuple_collection(internal::tuple_flatten(std::tuple {descriptor::get_collection_of(Cs{})...}));
    }


    static constexpr auto
    is_euclidean(const T&)
    {
      return std::bool_constant<(... and descriptor::euclidean_vector_space_descriptor<Cs>)>{};
    }


    static constexpr auto
    type_index(const T& t)
    {
      if constexpr (sizeof...(Cs) == 1) return descriptor::get_type_index(Cs{}...);
      else return std::type_index{typeid(T)};
    }


  private:

    // ------- Index tables ------- //

    /*
     * \internal
     * \tparam euclidean Whether the relevant vector is in Euclidean space (true) or not (false)
     * \tparam i The row index
     * \tparam t The component index within the list of descriptors
     * \tparam local_index The local index for indices associated with each of descriptors (resets to 0 when t increments)
     * \tparam start The start location in the corresponding euclidean or non-euclidean vector
     * \return An array of arrays of {t, local_index, start}
     */
    template<bool euclidean, std::size_t i, std::size_t t, std::size_t local_index, std::size_t start, typename...Arrs>
    static constexpr auto make_table(Arrs&&...arrs)
    {
      if constexpr (t < sizeof...(Cs))
      {
        using C = typename descriptor::StaticDescriptor<Cs...>::template Select<t>;
        constexpr auto i_size = descriptor::dimension_size_of_v<C>;
        constexpr auto e_size = descriptor::euclidean_dimension_size_of_v<C>;
        if constexpr (local_index >= (euclidean ? e_size : i_size))
        {
          return make_table<euclidean, i, t + 1, 0, start + (euclidean ? i_size : e_size)>(std::forward<Arrs>(arrs)...);
        }
        else
        {
          return make_table<euclidean, i + 1, t, local_index + 1, start>(
            std::forward<Arrs>(arrs)..., std::array<std::size_t, 3> {t, local_index, start});
        }
      }
      else
      {
        return std::array<std::array<std::size_t, 3>, sizeof...(Arrs)> {std::forward<Arrs>(arrs)...};
      }
    }


    static constexpr auto index_table = make_table<false, 0, 0, 0, 0>();


    static constexpr auto euclidean_index_table = make_table<true, 0, 0, 0, 0>();


    // ------- Function tables ------- //


    template<typename Scalar>
    using GArr = std::array<Scalar(*)(const std::function<Scalar(std::size_t)>&, std::size_t, std::size_t),
      1 + sizeof...(Cs)>;


    template<typename Scalar>
    using SArr = std::array<void(*)(const std::function<void(const Scalar&, std::size_t)>&,
      const std::function<Scalar(std::size_t)>&, const Scalar&, std::size_t, std::size_t), 1 + sizeof...(Cs)>;


    template<typename Scalar> static constexpr GArr<Scalar>
    to_euclidean_array { [](const auto& g, std::size_t euclidean_local_index, std::size_t start)
      { return descriptor::to_euclidean_element(Cs{}, g, euclidean_local_index, start); }... };

    template<typename Scalar> static constexpr GArr<Scalar>
    from_euclidean_array { [](const auto& g, std::size_t local_index, std::size_t euclidean_start)
      { return descriptor::from_euclidean_element(Cs{}, g, local_index, euclidean_start); }... };

    template<typename Scalar> static constexpr GArr<Scalar>
    wrap_get_array { [](const auto& g, std::size_t local_index, std::size_t start)
      { return descriptor::get_wrapped_component(Cs{}, g, local_index, start); }... };

    template<typename Scalar> static constexpr SArr<Scalar>
    wrap_set_array { [](const auto& s, const auto& g, const Scalar& x, std::size_t local_index, std::size_t start)
      { return descriptor::set_wrapped_component(Cs{}, s, g, x, local_index, start); }... };

    static constexpr bool euclidean_type = (descriptor::euclidean_vector_space_descriptor<Cs> and ...);

  public:

#ifdef __cpp_concepts
    static constexpr value::value auto
    to_euclidean_component(const T&, const auto& g, const value::index auto& euclidean_local_index, const value::index auto& start)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type> and value::index<L> and value::index<S>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T&, const Getter& g, const L& euclidean_local_index, const S& start)
#endif
    {
      using Scalar = decltype(g(std::declval<std::size_t>()));
      auto [tp, comp_euclidean_local_index, comp_start] = euclidean_index_table[static_cast<std::size_t>(euclidean_local_index)];
      return to_euclidean_array<Scalar>[tp](g, comp_euclidean_local_index, static_cast<std::size_t>(start) + comp_start);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    from_euclidean_component(const T&, const auto& g, const value::index auto& local_index, const value::index auto& euclidean_start)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T&, const Getter& g, const L& local_index, const S& euclidean_start)
#endif
    {
      using Scalar = decltype(g(std::declval<std::size_t>()));
      auto [tp, comp_local_index, comp_euclidean_start] = index_table[static_cast<std::size_t>(local_index)];
      return from_euclidean_array<Scalar>[tp](g, comp_local_index, static_cast<std::size_t>(euclidean_start) + comp_euclidean_start);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    get_wrapped_component(const T&, const auto& g, const value::index auto& local_index, const value::index auto& start)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T&, const Getter& g, const L& local_index, const S& start)
#endif
    {
      using Scalar = decltype(g(std::declval<std::size_t>()));
      auto [tp, comp_local_index, comp_euclidean_start] = index_table[static_cast<std::size_t>(local_index)];
      return wrap_get_array<Scalar>[tp](g, comp_local_index, static_cast<std::size_t>(start) + static_cast<std::size_t>(local_index) - comp_local_index);
    }


#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T&, const auto& s, const auto& g, const value::value auto& x,
      const value::index auto& local_index, const value::index auto& start)
    requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
    template<typename Setter, typename Getter, typename X, typename L, typename S, std::enable_if_t<
      value::value<X> and value::index<L> and value::index<S> and
      std::is_invocable<const Setter&, const X&, std::size_t>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T&, const Setter& s, const Getter& g, const X& x, const L& local_index, const S& start)
#endif
    {
      auto [tp, comp_local_index, comp_euclidean_start] = index_table[static_cast<std::size_t>(local_index)];
      wrap_set_array<std::decay_t<decltype(x)>>[tp](s, g, x, comp_local_index, static_cast<std::size_t>(start) + static_cast<std::size_t>(local_index) - comp_local_index);
    }

  };


}// namespace OpenKalman::interface


#endif //OPENKALMAN_STATICDESCRIPTOR_HPP
